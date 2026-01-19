"""Claude Code CLI LLM backend implementation.

This module provides an LLM backend that uses the Claude Code CLI (`claude`)
to generate responses. It allows EvoAgentX's self-evolution features to
leverage Claude's agentic capabilities.
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
from .claude_configs import ClaudeCodeLLMConfig
from .model_utils import Cost, cost_manager


# Cost per token for Claude models (approximate, may vary)
CLAUDE_MODEL_COSTS = {
    "sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},  # Claude 3.5 Sonnet
    "opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},    # Claude 3 Opus
    "haiku": {"input": 0.00025 / 1000, "output": 0.00125 / 1000},  # Claude 3 Haiku
}


def get_claude_model_cost():
    """Get the cost dictionary for Claude models."""
    return CLAUDE_MODEL_COSTS


@register_model(config_cls=ClaudeCodeLLMConfig, alias=["claude_code", "claude_cli"])
class ClaudeCodeLLM(BaseLLM):
    """LLM backend using Claude Code CLI.

    This class implements the BaseLLM interface using the `claude` CLI command.
    It supports both single and batch generation, with cost tracking.

    Example:
        ```python
        from evoagentx.models import ClaudeCodeLLM, ClaudeCodeLLMConfig

        config = ClaudeCodeLLMConfig(
            model="sonnet",
            timeout=60,
            max_budget_usd=0.10
        )
        llm = ClaudeCodeLLM(config)
        result = llm.generate(prompt="What is 2+2?")
        print(result)  # "4"
        ```
    """

    def init_model(self):
        """Initialize the Claude Code CLI backend.

        Verifies that the `claude` CLI is installed and accessible.

        Raises:
            RuntimeError: If the `claude` CLI is not found.
        """
        config: ClaudeCodeLLMConfig = self.config

        # Check if claude CLI is available
        self._claude_path = shutil.which("claude")
        if self._claude_path is None:
            raise RuntimeError(
                "Claude Code CLI not found. Please install it with: "
                "npm install -g @anthropic-ai/claude-code"
            )

        # Validate model
        valid_models = ["sonnet", "opus", "haiku"]
        if config.model not in valid_models:
            raise ValueError(
                f"Invalid model '{config.model}'. Must be one of: {valid_models}"
            )

        # Default ignore fields for parameter extraction
        self._default_ignore_fields = [
            "llm_type", "output_response", "verbose"
        ]

        logger.info(f"Initialized ClaudeCodeLLM with model={config.model}")

    def formulate_messages(
        self,
        prompts: List[str],
        system_messages: Optional[List[str]] = None
    ) -> List[List[dict]]:
        """Convert prompts to message format.

        For Claude Code CLI, we combine system message and prompt into a single
        prompt string, as the CLI doesn't support separate system messages directly.

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
        """Build the CLI command for Claude Code.

        Args:
            prompt: The user prompt.
            system_message: Optional system message.
            **kwargs: Additional parameters.

        Returns:
            List of command arguments.
        """
        config: ClaudeCodeLLMConfig = self.config
        cmd = [self._claude_path]

        # Add prompt
        cmd.extend(["-p", prompt])

        # Add model
        cmd.extend(["--model", config.model])

        # Add output format
        if config.output_format == "json":
            cmd.append("--output-format=json")
        elif config.output_format == "stream-json":
            cmd.append("--output-format=stream-json")

        # Add system prompt if provided
        effective_system = kwargs.get("system_message", system_message or config.system_prompt)
        if effective_system:
            cmd.extend(["--system-prompt", effective_system])

        # Add permission mode
        if config.permission_mode == "acceptEdits":
            cmd.append("--dangerously-skip-permissions")
        elif config.permission_mode == "bypassPermissions":
            cmd.append("--dangerously-skip-permissions")

        # Add max turns if specified
        max_turns = kwargs.get("max_turns", config.max_turns)
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])

        # Add allowed tools
        for tool in config.allowed_tools:
            cmd.extend(["--allowedTools", tool])

        # Add disallowed tools
        for tool in config.disallowed_tools:
            cmd.extend(["--disallowedTools", tool])

        # Print mode for non-interactive use
        cmd.append("--print")

        return cmd

    def _parse_cli_output(self, stdout: str, stderr: str, output_format: str) -> str:
        """Parse the CLI output.

        Args:
            stdout: Standard output from the CLI.
            stderr: Standard error from the CLI.
            output_format: The expected output format.

        Returns:
            The parsed response text.

        Raises:
            RuntimeError: If parsing fails.
        """
        if output_format == "json":
            try:
                data = json.loads(stdout)
                # Extract the response from JSON structure
                if isinstance(data, dict):
                    if "result" in data:
                        return data["result"]
                    if "response" in data:
                        return data["response"]
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
        costs = CLAUDE_MODEL_COSTS.get(model, CLAUDE_MODEL_COSTS["sonnet"])
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
        cost_manager.update_cost(cost=cost, model=f"claude-{self.config.model}")

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
        config: ClaudeCodeLLMConfig = self.config
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
                    f"Claude Code CLI failed with exit code {result.returncode}: {error_msg}"
                )

            output = self._parse_cli_output(
                result.stdout,
                result.stderr,
                config.output_format
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
                f"Claude Code CLI timed out after {config.timeout} seconds"
            )
        except Exception as e:
            raise RuntimeError(f"Error during Claude Code CLI execution: {str(e)}")

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
        config: ClaudeCodeLLMConfig = self.config
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
                    f"Claude Code CLI timed out after {config.timeout} seconds"
                )

            if process.returncode != 0:
                error_msg = stderr.decode() or stdout.decode()
                raise RuntimeError(
                    f"Claude Code CLI failed with exit code {process.returncode}: {error_msg}"
                )

            output = self._parse_cli_output(
                stdout.decode(),
                stderr.decode(),
                config.output_format
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
            raise RuntimeError(f"Error during async Claude Code CLI execution: {str(e)}")


__all__ = ["ClaudeCodeLLM", "get_claude_model_cost"]
