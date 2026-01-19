"""Configuration classes for Claude Code CLI and Codex CLI backends.

These configurations enable EvoAgentX to use Claude Code CLI and Codex CLI
as LLM backends, allowing the self-evolution features to work with
agentic systems.
"""

from pydantic import Field
from typing import Optional, Union, List

from .model_configs import LLMConfig


class ClaudeCodeLLMConfig(LLMConfig):
    """Configuration for Claude Code CLI backend.

    This config enables using the `claude` CLI command as an LLM backend.
    Claude Code is Anthropic's official CLI for Claude, providing access
    to Claude models with tool use capabilities.

    Attributes:
        llm_type: Identifier for this LLM type.
        model: The Claude model to use (sonnet, opus, haiku).
        permission_mode: Permission handling mode (default, acceptEdits, bypassPermissions).
        output_format: Output format (text, json, stream-json).
        timeout: Timeout in seconds for CLI execution.
        working_directory: Working directory for Claude Code operations.
        system_prompt: System prompt to prepend to all requests.
        allowed_tools: List of tools to allow (empty = all allowed).
        disallowed_tools: List of tools to disallow.
        max_turns: Maximum number of agentic turns.
        max_budget_usd: Maximum cost budget in USD.
        verbose: Enable verbose output for debugging.
    """

    llm_type: str = "ClaudeCodeLLM"
    model: str = Field(default="sonnet", description="The Claude model to use: sonnet, opus, or haiku")

    # Permission and safety settings
    permission_mode: str = Field(
        default="default",
        description="Permission handling: default, acceptEdits, bypassPermissions"
    )

    # Output settings
    output_format: str = Field(
        default="text",
        description="Output format: text, json, or stream-json"
    )

    # Execution settings
    timeout: int = Field(
        default=300,
        description="Timeout in seconds for CLI execution"
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Working directory for Claude Code operations"
    )

    # Prompt settings
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend to all requests"
    )

    # Tool settings
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="List of tools to allow (empty = all allowed)"
    )
    disallowed_tools: List[str] = Field(
        default_factory=list,
        description="List of tools to disallow"
    )

    # Agentic settings
    max_turns: Optional[int] = Field(
        default=None,
        description="Maximum number of agentic turns"
    )

    # Cost control
    max_budget_usd: Optional[float] = Field(
        default=None,
        description="Maximum cost budget in USD"
    )

    # Debug settings
    verbose: bool = Field(
        default=False,
        description="Enable verbose output for debugging"
    )

    def __str__(self) -> str:
        return f"ClaudeCode({self.model})"


class CodexLLMConfig(LLMConfig):
    """Configuration for Codex CLI backend.

    This config enables using the `codex` CLI command as an LLM backend.
    Codex is OpenAI's CLI tool for code generation and execution.

    Attributes:
        llm_type: Identifier for this LLM type.
        model: The model to use (o3, gpt-4.1, etc.).
        sandbox: Sandbox mode (read-only, workspace-write, danger-full-access).
        approval_policy: Approval policy (untrusted, on-failure, on-request, never).
        timeout: Timeout in seconds for CLI execution.
        json_output: Whether to use JSON output mode.
        working_directory: Working directory for Codex operations.
        base_instructions: Base instructions to override defaults.
        developer_instructions: Developer-role instructions.
        profile: Configuration profile from config.toml.
        verbose: Enable verbose output for debugging.
    """

    llm_type: str = "CodexLLM"
    model: str = Field(
        default="o3",
        description="The model to use: o3, gpt-4.1, etc."
    )

    # Sandbox and approval settings
    sandbox: str = Field(
        default="read-only",
        description="Sandbox mode: read-only, workspace-write, danger-full-access"
    )
    approval_policy: str = Field(
        default="untrusted",
        description="Approval policy: untrusted, on-failure, on-request, never"
    )

    # Execution settings
    timeout: int = Field(
        default=300,
        description="Timeout in seconds for CLI execution"
    )
    json_output: bool = Field(
        default=True,
        description="Whether to use JSON output mode"
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Working directory for Codex operations"
    )

    # Instruction settings
    base_instructions: Optional[str] = Field(
        default=None,
        description="Base instructions to override defaults"
    )
    developer_instructions: Optional[str] = Field(
        default=None,
        description="Developer-role instructions"
    )

    # Profile settings
    profile: Optional[str] = Field(
        default=None,
        description="Configuration profile from config.toml"
    )

    # Debug settings
    verbose: bool = Field(
        default=False,
        description="Enable verbose output for debugging"
    )

    def __str__(self) -> str:
        return f"Codex({self.model})"


class CodexMCPLLMConfig(LLMConfig):
    """Configuration for Codex MCP Server backend.

    This config enables using the Codex MCP Server as an LLM backend,
    allowing integration through the Model Context Protocol.

    Attributes:
        llm_type: Identifier for this LLM type.
        model: The model to use.
        server_url: URL of the Codex MCP server.
        sandbox: Sandbox mode.
        approval_policy: Approval policy.
        timeout: Timeout in seconds for requests.
        working_directory: Working directory for operations.
        base_instructions: Base instructions to override defaults.
        developer_instructions: Developer-role instructions.
    """

    llm_type: str = "CodexMCPLLM"
    model: str = Field(
        default="o3",
        description="The model to use"
    )

    # MCP Server settings
    server_url: Optional[str] = Field(
        default=None,
        description="URL of the Codex MCP server (if using remote server)"
    )

    # Sandbox and approval settings
    sandbox: str = Field(
        default="read-only",
        description="Sandbox mode: read-only, workspace-write, danger-full-access"
    )
    approval_policy: str = Field(
        default="untrusted",
        description="Approval policy: untrusted, on-failure, on-request, never"
    )

    # Execution settings
    timeout: int = Field(
        default=300,
        description="Timeout in seconds for requests"
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Working directory for operations"
    )

    # Instruction settings
    base_instructions: Optional[str] = Field(
        default=None,
        description="Base instructions to override defaults"
    )
    developer_instructions: Optional[str] = Field(
        default=None,
        description="Developer-role instructions"
    )

    def __str__(self) -> str:
        return f"CodexMCP({self.model})"


__all__ = [
    "ClaudeCodeLLMConfig",
    "CodexLLMConfig",
    "CodexMCPLLMConfig",
]
