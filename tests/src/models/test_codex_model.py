"""Tests for Codex CLI LLM backend."""

import pytest
import subprocess
from unittest.mock import MagicMock, patch
import json

from evoagentx.models import CodexLLMConfig, CodexLLM
from evoagentx.models import LLMOutputParser
from evoagentx.models import cost_manager


class MockCompletedProcess:
    """Mock subprocess.CompletedProcess for testing."""

    def __init__(self, stdout="Test response", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def mock_subprocess_run(*args, **kwargs):
    """Mock subprocess.run for Codex CLI."""
    return MockCompletedProcess(
        stdout=json.dumps({"result": "def sort_list(lst): return sorted(lst)"}),
        stderr="",
        returncode=0
    )


def mock_subprocess_run_text(*args, **kwargs):
    """Mock subprocess.run with text output."""
    return MockCompletedProcess(
        stdout="def sort_list(lst): return sorted(lst)",
        stderr="",
        returncode=0
    )


def mock_shutil_which(cmd):
    """Mock shutil.which to return a fake path."""
    if cmd == "codex":
        return "/usr/local/bin/codex"
    return None


@pytest.fixture
def codex_config():
    """Create a test configuration."""
    return CodexLLMConfig(
        model="o3",
        output_response=False,
        timeout=60,
        json_output=True
    )


@pytest.fixture
def codex_config_text():
    """Create a test configuration without JSON output."""
    return CodexLLMConfig(
        model="o3",
        output_response=False,
        timeout=60,
        json_output=False
    )


def test_codex_config():
    """Test CodexLLMConfig defaults and validation."""
    config = CodexLLMConfig(model="o3")

    assert config.llm_type == "CodexLLM"
    assert config.model == "o3"
    assert config.sandbox == "read-only"
    assert config.approval_policy == "untrusted"
    assert config.timeout == 300
    assert config.json_output == True
    assert str(config) == "Codex(o3)"


def test_codex_config_all_models():
    """Test various Codex models."""
    for model in ["o3", "o3-mini", "gpt-4.1", "gpt-4.1-mini"]:
        config = CodexLLMConfig(model=model)
        assert config.model == model


def test_codex_config_sandbox_modes():
    """Test sandbox mode configurations."""
    for sandbox in ["read-only", "workspace-write", "danger-full-access"]:
        config = CodexLLMConfig(model="o3", sandbox=sandbox)
        assert config.sandbox == sandbox


def test_codex_config_approval_policies():
    """Test approval policy configurations."""
    for policy in ["untrusted", "on-failure", "on-request", "never"]:
        config = CodexLLMConfig(model="o3", approval_policy=policy)
        assert config.approval_policy == policy


@patch("shutil.which", mock_shutil_which)
def test_codex_init(codex_config):
    """Test CodexLLM initialization."""
    llm = CodexLLM(codex_config)
    assert llm._codex_path == "/usr/local/bin/codex"


@patch("shutil.which", lambda x: None)
def test_codex_init_no_cli():
    """Test error when CLI is not found."""
    config = CodexLLMConfig(model="o3")
    with pytest.raises(RuntimeError, match="Codex CLI not found"):
        CodexLLM(config)


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run)
def test_codex_generation(codex_config):
    """Test basic generation."""
    llm = CodexLLM(codex_config)

    prompt = "Write a function to sort a list"
    system_prompt = "You are an expert Python programmer"

    output = llm.generate(prompt=prompt, system_message=system_prompt)
    assert isinstance(output, LLMOutputParser)
    assert "sort" in output.content.lower()


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run_text)
def test_codex_text_output(codex_config_text):
    """Test generation with text output."""
    llm = CodexLLM(codex_config_text)

    prompt = "Write a function to sort a list"
    output = llm.generate(prompt=prompt)
    assert isinstance(output, LLMOutputParser)
    assert "sort" in output.content.lower()


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run)
def test_codex_batch_generation(codex_config):
    """Test batch generation."""
    llm = CodexLLM(codex_config)

    prompts = ["Write a function", "Write another function"]
    outputs = llm.generate(prompt=prompts)

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    for output in outputs:
        assert isinstance(output, LLMOutputParser)


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run", mock_subprocess_run)
def test_codex_formulate_messages(codex_config):
    """Test message formatting."""
    llm = CodexLLM(codex_config)

    prompts = ["Hello"]
    system_messages = ["You are helpful"]

    messages = llm.formulate_messages(prompts, system_messages)

    assert len(messages) == 1
    assert messages[0][0]["role"] == "system"
    assert messages[0][0]["content"] == "You are helpful"
    assert messages[0][1]["role"] == "user"
    assert messages[0][1]["content"] == "Hello"


@patch("shutil.which", mock_shutil_which)
def test_codex_build_cli_command(codex_config):
    """Test CLI command building."""
    llm = CodexLLM(codex_config)

    cmd = llm._build_cli_command("Test prompt", "System message")

    assert "/usr/local/bin/codex" in cmd
    assert "exec" in cmd
    assert "Test prompt" in cmd
    assert "--model" in cmd
    assert "o3" in cmd
    assert "--sandbox" in cmd
    assert "read-only" in cmd
    assert "--approval-policy" in cmd
    assert "untrusted" in cmd
    assert "--developer-instructions" in cmd
    assert "System message" in cmd
    assert "--json" in cmd


@patch("shutil.which", mock_shutil_which)
def test_codex_estimate_tokens(codex_config):
    """Test token estimation."""
    llm = CodexLLM(codex_config)

    # ~4 characters per token
    text = "a" * 100
    tokens = llm._estimate_tokens(text)
    assert tokens == 25


@patch("shutil.which", mock_shutil_which)
def test_codex_compute_cost(codex_config):
    """Test cost computation."""
    llm = CodexLLM(codex_config)

    cost = llm._compute_cost(input_tokens=100, output_tokens=50)

    assert cost.input_tokens == 100
    assert cost.output_tokens == 50
    assert cost.input_cost > 0
    assert cost.output_cost > 0


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run")
def test_codex_timeout(mock_run, codex_config):
    """Test timeout handling."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=60)

    llm = CodexLLM(codex_config)

    with pytest.raises(RuntimeError, match="timed out"):
        llm.generate(prompt="Test")


@patch("shutil.which", mock_shutil_which)
@patch("subprocess.run")
def test_codex_cli_error(mock_run, codex_config):
    """Test CLI error handling."""
    mock_run.return_value = MockCompletedProcess(
        stdout="",
        stderr="Error: Invalid prompt",
        returncode=1
    )

    llm = CodexLLM(codex_config)

    with pytest.raises(RuntimeError, match="Codex CLI failed"):
        llm.generate(prompt="Test")


@patch("shutil.which", mock_shutil_which)
def test_codex_config_with_instructions(codex_config):
    """Test config with base and developer instructions."""
    config = CodexLLMConfig(
        model="o3",
        base_instructions="Be concise",
        developer_instructions="Return JSON",
        profile="custom-profile"
    )

    assert config.base_instructions == "Be concise"
    assert config.developer_instructions == "Return JSON"
    assert config.profile == "custom-profile"
