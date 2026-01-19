"""Tests for Claude Code LLM backend."""

import pytest
import subprocess
from unittest.mock import MagicMock, patch

from evoagentx.models import ClaudeCodeLLMConfig, ClaudeCodeLLM
from evoagentx.models import LLMOutputParser
from evoagentx.models import cost_manager


class MockCompletedProcess:
    """Mock subprocess.CompletedProcess for testing."""

    def __init__(self, stdout="Test response", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def mock_subprocess_run(*args, **kwargs):
    """Mock subprocess.run for Claude CLI."""
    return MockCompletedProcess(stdout="Beijing", stderr="", returncode=0)


def mock_subprocess_run_json(*args, **kwargs):
    """Mock subprocess.run with JSON output."""
    import json
    return MockCompletedProcess(
        stdout=json.dumps({"result": "Beijing"}),
        stderr="",
        returncode=0
    )


def mock_shutil_which(cmd):
    """Mock shutil.which to return a fake path."""
    if cmd == "claude":
        return "/usr/local/bin/claude"
    return None


@pytest.fixture
def claude_config():
    """Create a test configuration."""
    return ClaudeCodeLLMConfig(
        model="sonnet",
        output_response=False,
        timeout=60
    )


@pytest.fixture
def claude_config_json():
    """Create a test configuration with JSON output."""
    return ClaudeCodeLLMConfig(
        model="sonnet",
        output_format="json",
        output_response=False,
        timeout=60
    )


def test_claude_code_config():
    """Test ClaudeCodeLLMConfig defaults and validation."""
    config = ClaudeCodeLLMConfig(model="sonnet")

    assert config.llm_type == "ClaudeCodeLLM"
    assert config.model == "sonnet"
    assert config.permission_mode == "default"
    assert config.output_format == "text"
    assert config.timeout == 300
    assert str(config) == "ClaudeCode(sonnet)"


def test_claude_code_config_all_models():
    """Test all supported Claude models."""
    for model in ["sonnet", "opus", "haiku"]:
        config = ClaudeCodeLLMConfig(model=model)
        assert config.model == model


@patch("shutil.which", mock_shutil_which)
def test_claude_code_init(claude_config):
    """Test ClaudeCodeLLM initialization."""
    llm = ClaudeCodeLLM(claude_config)
    assert llm._claude_path == "/usr/local/bin/claude"


@patch("shutil.which", lambda x: None)
def test_claude_code_init_no_cli():
    """Test error when CLI is not found."""
    config = ClaudeCodeLLMConfig(model="sonnet")
    with pytest.raises(RuntimeError, match="Claude Code CLI not found"):
        ClaudeCodeLLM(config)


@patch("shutil.which", mock_shutil_which)
def test_claude_code_invalid_model():
    """Test error with invalid model."""
    config = ClaudeCodeLLMConfig(model="invalid-model")
    with pytest.raises(ValueError, match="Invalid model"):
        ClaudeCodeLLM(config)


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run)
def test_claude_code_generation(claude_config):
    """Test basic generation."""
    llm = ClaudeCodeLLM(claude_config)

    prompt = "What is the capital of China?"
    system_prompt = "You are an expert in geography"

    output = llm.generate(prompt=prompt, system_message=system_prompt)
    assert isinstance(output, LLMOutputParser)
    assert output.content == "Beijing"


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run_json)
def test_claude_code_json_output(claude_config_json):
    """Test generation with JSON output format."""
    llm = ClaudeCodeLLM(claude_config_json)

    prompt = "What is the capital of China?"
    output = llm.generate(prompt=prompt)
    assert isinstance(output, LLMOutputParser)
    assert output.content == "Beijing"


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run)
def test_claude_code_batch_generation(claude_config):
    """Test batch generation."""
    llm = ClaudeCodeLLM(claude_config)

    prompts = ["What is 2+2?", "What is 3+3?"]
    outputs = llm.generate(prompt=prompts)

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    for output in outputs:
        assert isinstance(output, LLMOutputParser)


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run)
def test_claude_code_formulate_messages(claude_config):
    """Test message formatting."""
    llm = ClaudeCodeLLM(claude_config)

    prompts = ["Hello"]
    system_messages = ["You are helpful"]

    messages = llm.formulate_messages(prompts, system_messages)

    assert len(messages) == 1
    assert messages[0][0]["role"] == "system"
    assert messages[0][0]["content"] == "You are helpful"
    assert messages[0][1]["role"] == "user"
    assert messages[0][1]["content"] == "Hello"


@patch("shutil.which", mock_shutil_which)
def test_claude_code_build_cli_command(claude_config):
    """Test CLI command building."""
    llm = ClaudeCodeLLM(claude_config)

    cmd = llm._build_cli_command("Test prompt", "System message")

    assert "/usr/local/bin/claude" in cmd
    assert "-p" in cmd
    assert "Test prompt" in cmd
    assert "--model" in cmd
    assert "sonnet" in cmd
    assert "--system-prompt" in cmd
    assert "System message" in cmd
    assert "--print" in cmd


@patch("shutil.which", mock_shutil_which)
def test_claude_code_estimate_tokens(claude_config):
    """Test token estimation."""
    llm = ClaudeCodeLLM(claude_config)

    # ~4 characters per token
    text = "a" * 100
    tokens = llm._estimate_tokens(text)
    assert tokens == 25


@patch("shutil.which", mock_shutil_which)
def test_claude_code_compute_cost(claude_config):
    """Test cost computation."""
    llm = ClaudeCodeLLM(claude_config)

    cost = llm._compute_cost(input_tokens=100, output_tokens=50)

    assert cost.input_tokens == 100
    assert cost.output_tokens == 50
    assert cost.input_cost > 0
    assert cost.output_cost > 0


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run")
def test_claude_code_timeout(mock_run, claude_config):
    """Test timeout handling."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)

    llm = ClaudeCodeLLM(claude_config)

    with pytest.raises(RuntimeError, match="timed out"):
        llm.generate(prompt="Test")


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run")
def test_claude_code_cli_error(mock_run, claude_config):
    """Test CLI error handling."""
    mock_run.return_value = MockCompletedProcess(
        stdout="",
        stderr="Error: Invalid prompt",
        returncode=1
    )

    llm = ClaudeCodeLLM(claude_config)

    with pytest.raises(RuntimeError, match="Claude Code CLI failed"):
        llm.generate(prompt="Test")
