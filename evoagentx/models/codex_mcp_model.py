"""Codex MCP Server LLM backend implementation.

This module provides an LLM backend that uses the Codex MCP Server
to generate responses. It allows EvoAgentX's self-evolution features to
leverage OpenAI's Codex through the Model Context Protocol.
"""

import os
import json
import asyncio
from typing import Optional, List, Any, Dict

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from ..core.registry import register_model
from ..core.logging import logger
from .base_model import BaseLLM
from .claude_configs import CodexMCPLLMConfig
from .model_utils import Cost, cost_manager
from .codex_model import CODEX_MODEL_COSTS, get_codex_model_cost


@register_model(config_cls=CodexMCPLLMConfig, alias=["codex_mcp"])
class CodexMCPLLM(BaseLLM):
    """LLM backend using Codex MCP Server.

    This class implements the BaseLLM interface using the Codex MCP Server.
    It communicates with the MCP server to execute prompts and get responses.

    Note: This implementation requires the mcp package to be installed.

    Example:
        ```python
        from evoagentx.models import CodexMCPLLM, CodexMCPLLMConfig

        config = CodexMCPLLMConfig(
            model="o3",
            sandbox="read-only",
            timeout=60
        )
        llm = CodexMCPLLM(config)
        result = llm.generate(prompt="Explain how to sort a list in Python")
        print(result)
        ```
    """

    def init_model(self):
        """Initialize the Codex MCP backend.

        Attempts to import the MCP client library.

        Raises:
            RuntimeError: If the MCP library is not available.
        """
        config: CodexMCPLLMConfig = self.config

        # Try to import MCP client
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            self._mcp_available = True
            self._ClientSession = ClientSession
            self._StdioServerParameters = StdioServerParameters
            self._stdio_client = stdio_client
        except ImportError:
            self._mcp_available = False
            logger.warning(
                "MCP library not available. Install with: pip install mcp"
            )

        # Default ignore fields for parameter extraction
        self._default_ignore_fields = [
            "llm_type", "output_response", "server_url"
        ]

        logger.info(f"Initialized CodexMCPLLM with model={config.model}")

    def formulate_messages(
        self,
        prompts: List[str],
        system_messages: Optional[List[str]] = None
    ) -> List[List[dict]]:
        """Convert prompts to message format.

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

    def _build_mcp_request(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build the MCP request for Codex.

        Args:
            prompt: The user prompt.
            system_message: Optional system message.
            **kwargs: Additional parameters.

        Returns:
            Dictionary with MCP request parameters.
        """
        config: CodexMCPLLMConfig = self.config

        request = {
            "prompt": prompt,
            "model": config.model,
            "sandbox": config.sandbox,
            "approval-policy": config.approval_policy,
        }

        # Add developer instructions
        effective_system = kwargs.get("system_message", system_message or config.developer_instructions)
        if effective_system:
            request["developer-instructions"] = effective_system

        # Add base instructions if specified
        base_instructions = kwargs.get("base_instructions", config.base_instructions)
        if base_instructions:
            request["base-instructions"] = base_instructions

        # Add working directory
        cwd = kwargs.get("working_directory", config.working_directory)
        if cwd:
            request["cwd"] = cwd

        return request

    def _parse_mcp_response(self, response: Any) -> str:
        """Parse the MCP response.

        Args:
            response: The response from the MCP server.

        Returns:
            The parsed response text.
        """
        if isinstance(response, dict):
            # Try common response fields
            for key in ["result", "response", "output", "content", "message", "text"]:
                if key in response:
                    value = response[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict) and "text" in value:
                        return value["text"]
            # Return JSON string if no known field found
            return json.dumps(response)
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

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
        cost_manager.update_cost(cost=cost, model=f"codex-mcp-{self.config.model}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """Generate a response for a single set of messages.

        For synchronous generation, this wraps the async implementation.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters.

        Returns:
            The generated response text.

        Raises:
            RuntimeError: If MCP is not available or execution fails.
        """
        if not self._mcp_available:
            raise RuntimeError(
                "MCP library not available. Install with: pip install mcp"
            )

        # Run the async version synchronously
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context, use run_in_executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.single_generate_async(messages, **kwargs)
                )
                return future.result(timeout=self.config.timeout)
        else:
            return loop.run_until_complete(
                self.single_generate_async(messages, **kwargs)
            )

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
        """Asynchronously generate a response using MCP.

        Args:
            messages: List of message dicts.
            **kwargs: Additional generation parameters.

        Returns:
            The generated response text.
        """
        if not self._mcp_available:
            raise RuntimeError(
                "MCP library not available. Install with: pip install mcp"
            )

        config: CodexMCPLLMConfig = self.config
        output_response = kwargs.get("output_response", config.output_response)

        # Extract prompt and system message
        prompt = ""
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                prompt = msg["content"]

        # Build MCP request
        request = self._build_mcp_request(prompt, system_message, **kwargs)

        try:
            # Connect to MCP server and execute
            server_params = self._StdioServerParameters(
                command="npx",
                args=["-y", "@openai/codex-mcp"],
                env=None
            )

            async with self._stdio_client(server_params) as (read, write):
                async with self._ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # Call the codex tool
                    result = await asyncio.wait_for(
                        session.call_tool("codex", request),
                        timeout=config.timeout
                    )

                    output = self._parse_mcp_response(result)

            # Estimate and track costs
            input_tokens = self._estimate_tokens(prompt + (system_message or ""))
            output_tokens = self._estimate_tokens(output)
            cost = self._compute_cost(input_tokens, output_tokens)
            self._update_cost(cost)

            if output_response:
                print(output)

            return output

        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Codex MCP request timed out after {config.timeout} seconds"
            )
        except Exception as e:
            raise RuntimeError(f"Error during Codex MCP execution: {str(e)}")


__all__ = ["CodexMCPLLM"]
