"""Codex CLI LLM backend implementation.

This module provides an LLM backend that uses the Codex CLI (`codex`)
to generate responses. It allows EvoAgentX's self-evolution features to
leverage OpenAI's Codex agentic capabilities.
"""

import os
import json
import asyncio
import subprocess
import shutil
from typing import Optional, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from ..core.registry import register_model
from ..core.logging import logger
from .base_model import BaseLLM
from .claude_configs import CodexLLMConfig
from .model_utils import Cost, cost_manager


# Cost per token for Codex/OpenAI models (approximate)
CODEX_MODEL_COSTS = {
    "o3": {"input": 0.015 / 1000, "output": 0.060 / 1000},
    "o3-mini": {"input": 0.0011 / 1000, "output": 0.0044 / 1000},
    "gpt-4.1": {"input": 0.002 / 1000, "output": 0.008 / 1000},
    "gpt-4.1-mini": {"input": 0.0004 / 1000, "output": 0.0016 / 1000},
    "gpt-4.1-nano": {"input": 0.0001 / 1000, "output": 0.0004 / 1000},
}


def get_codex_model_cost():
    """Get the cost dictionary for Codex models."""
    return CODEX_MODEL_COSTS


@register_model(config_cls=CodexLLMConfig, alias=["codex_cli", "codex"])
class CodexLLM(BaseLLM):
    """LLM backend using Codex CLI.

    This class implements the BaseLLM interface using the `codex` CLI command.
    It supports both single and batch generation, with cost tracking.

    Example:
        ```python
        from evoagentx.models import CodexLLM, CodexLLMConfig

        config = CodexLLMConfig(
            model="o3",
            sandbox="read-only",
            timeout=60
        )
        llm = CodexLLM(config)
        result = llm.generate(prompt="Explain how to sort a list in Python")
        print(result)
        ```
    """

    def init_model(self):
        """Initialize the Codex CLI backend.

        Verifies that the `codex` CLI is installed and accessible.

        Raises:
            RuntimeError: If the `codex` CLI is not found.
        """
        config: CodexLLMConfig = self.config

        # Check if codex CLI is available
        self._codex_path = shutil.which("codex")
        if self._codex_path is None:
            raise RuntimeError(
                "Codex CLI not found. Please install it with: "
                "npm install -g @openai/codex"
            )

        # Default ignore fields for parameter extraction
        self._default_ignore_fields = [
            "llm_type", "output_response", "verbose", "json_output"
        ]

        logger.info(f"Initialized CodexLLM with model={config.model}")

    def formulate_messages(
        self,
        prompts: List[str],
        system_messages: Optional[List[str]] = None
    ) -> List[List[dict]]:
        """Convert prompts to message format.

        For Codex CLI, we combine system message and prompt into a single
        prompt string, as the CLI uses developer-instructions for system context.

        Args:
            prompts: List of user prompts.
            system_messages: Optional list of system messages.

        Returns:
            List of message lists in chat format.
        """
        if system_messages:
            assert len(prompts) == len(system_messages), (
                f"Number of prompts ({len(prompts)}) must match "
                f"number of system_messages ({len(system_messages)})"
            )
        else:
            system_messages = [None] * len(prompts)

        messages_list = []
        for prompt, system_message in zip(prompts, system_messages):
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        return messages_list

    def _build_cli_command(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Build the CLI command for Codex.

        Args:
            prompt: The user prompt.
            system_message: Optional system message (used as developer instructions).
            **kwargs: Additional parameters.

        Returns:
            List of command arguments.
        """
        config: CodexLLMConfig = self.config
        cmd = [self._codex_path]

        # Add the exec subcommand with prompt
        cmd.extend(["exec", prompt])

        # Add model
        cmd.extend(["--model", config.model])

        # Add sandbox mode
        cmd.extend(["--sandbox", config.sandbox])

        # Add approval policy
        cmd.extend(["--approval-policy", config.approval_policy])

        # Add developer instructions (system message)
        effective_system = kwargs.get("system_message", system_message or config.developer_instructions)
        if effective_system:
            cmd.extend(["--developer-instructions", effective_system])

        # Add base instructions if specified
        base_instructions = kwargs.get("base_instructions", config.base_instructions)
        if base_instructions:
            cmd.extend(["--base-instructions", base_instructions])

        # Add profile if specified
        profile = kwargs.get("profile", config.profile)
        if profile:
            cmd.extend(["--profile", profile])

        # Add JSON output flag if enabled
        if config.json_output:
            cmd.append("--json")

        return cmd

    def _parse_cli_output(self, stdout: str, stderr: str, json_output: bool) -> str:
        """Parse the CLI output.

        Args:
            stdout: Standard output from the CLI.
            stderr: Standard error from the CLI.
            json_output: Whether JSON output mode was used.

        Returns:
            The parsed response text.

        Raises:
            RuntimeError: If parsing fails.
        """
        if json_output:
            try:
                data = json.loads(stdout)
                # Extract the response from JSON structure
                if isinstance(data, dict):
                    if "result" in data:
                        return data["result"]
                    if "response" in data:
                        return data["response"]
                    if "output" in data:
                        return data["output"]
                    if "content" in data:
                        return data["content"]
                    if "message" in data:
                        return data["message"]
                return stdout
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                return stdout.strip()
        else:
            return stdout.strip()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> Cost:
        """Compute the cost for a generation.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost object with token counts and costs.
        """
        model = self.config.model
        # Default to o3 costs if model not found
        costs = CODEX_MODEL_COSTS.get(model, CODEX_MODEL_COSTS["o3"])
        input_cost = input_tokens * costs["input"]
        output_cost = output_tokens * costs["output"]

        return Cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost
        )

    def _update_cost(self, cost: Cost):
        """Update the cost manager with new costs.

        Args:
            cost: The cost to record.
        """
        cost_manager.update_cost(cost=cost, model=f"codex-{self.config.model}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """Generate a response for a single set of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters.

        Returns:
            The generated response text.

        Raises:
            RuntimeError: If CLI execution fails.
        """
        config: CodexLLMConfig = self.config
        output_response = kwargs.get("output_response", config.output_response)

        # Extract prompt and system message from messages
        prompt = ""
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                prompt = msg["content"]

        # Build command
        cmd = self._build_cli_command(prompt, system_message, **kwargs)

        # Set up environment
        env = os.environ.copy()

        # Set working directory
        cwd = kwargs.get("working_directory", config.working_directory)

        if config.verbose:
            logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            # Run CLI command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout,
                cwd=cwd,
                env=env
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                raise RuntimeError(
                    f"Codex CLI failed with exit code {result.returncode}: {error_msg}"
                )

            output = self._parse_cli_output(
                result.stdout,
                result.stderr,
                config.json_output
            )

            # Estimate and track costs
            input_tokens = self._estimate_tokens(prompt + (system_message or ""))
            output_tokens = self._estimate_tokens(output)
            cost = self._compute_cost(input_tokens, output_tokens)
            self._update_cost(cost)

            if output_response:
                print(output)

            return output

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Codex CLI timed out after {config.timeout} seconds"
            )
        except Exception as e:
            raise RuntimeError(f"Error during Codex CLI execution: {str(e)}")

    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """Generate responses for a batch of message sets.

        Args:
            batch_messages: List of message lists.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated responses.
        """
        return [
            self.single_generate(messages=messages, **kwargs)
            for messages in batch_messages
        ]

    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        """Asynchronously generate a response.

        Args:
            messages: List of message dicts.
            **kwargs: Additional generation parameters.

        Returns:
            The generated response text.
        """
        config: CodexLLMConfig = self.config
        output_response = kwargs.get("output_response", config.output_response)

        # Extract prompt and system message
        prompt = ""
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                prompt = msg["content"]

        # Build command
        cmd = self._build_cli_command(prompt, system_message, **kwargs)

        # Set working directory
        cwd = kwargs.get("working_directory", config.working_directory)

        if config.verbose:
            logger.debug(f"Running async command: {' '.join(cmd)}")

        try:
            # Run CLI command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise RuntimeError(
                    f"Codex CLI timed out after {config.timeout} seconds"
                )

            if process.returncode != 0:
                error_msg = stderr.decode() or stdout.decode()
                raise RuntimeError(
                    f"Codex CLI failed with exit code {process.returncode}: {error_msg}"
                )

            output = self._parse_cli_output(
                stdout.decode(),
                stderr.decode(),
                config.json_output
            )

            # Estimate and track costs
            input_tokens = self._estimate_tokens(prompt + (system_message or ""))
            output_tokens = self._estimate_tokens(output)
            cost = self._compute_cost(input_tokens, output_tokens)
            self._update_cost(cost)

            if output_response:
                print(output)

            return output

        except Exception as e:
            if "timed out" in str(e):
                raise
            raise RuntimeError(f"Error during async Codex CLI execution: {str(e)}")


__all__ = ["CodexLLM", "get_codex_model_cost"]
