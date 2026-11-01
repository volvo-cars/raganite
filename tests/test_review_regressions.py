"""Configuration and base-installation regressions identified during PR review."""

import subprocess
import sys
from pathlib import Path

import pytest
from benchmark.cli import main as benchmark_main
from click.testing import CliRunner

from raganite.cli import main
from raganite.config import loading


def test_explicit_missing_env_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        loading.load_config(env_file=tmp_path / "missing.env")


def test_default_dotenv_is_loaded_only_from_current_directory(tmp_path, monkeypatch):
    parent_env = tmp_path / ".env"
    parent_env.touch()
    working_directory = tmp_path / "child"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    calls = []
    monkeypatch.setattr(
        loading, "load_dotenv", lambda path, *, override: calls.append((path, override))
    )
    loading.load_config()
    assert calls == []
    local_env = working_directory / ".env"
    local_env.touch()
    loading.load_config()
    assert calls == [(local_env, False)]


def test_cli_loads_local_dotenv_and_preserves_process_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGANITE_NAMESPACE", raising=False)
    runner = CliRunner()
    Path(".env").write_text("RAGANITE_NAMESPACE=from-dotenv\n", encoding="utf-8")
    result = runner.invoke(main, ["config"])
    assert result.exit_code == 0, result.output
    assert '"namespace": "from-dotenv"' in result.output
    monkeypatch.setenv("RAGANITE_NAMESPACE", "from-process")
    result = runner.invoke(main, ["config"])
    assert result.exit_code == 0, result.output
    assert '"namespace": "from-process"' in result.output


def test_cli_supports_explicit_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGANITE_NAMESPACE", raising=False)
    runner = CliRunner()
    Path("selected.env").write_text("RAGANITE_NAMESPACE=selected\n", encoding="utf-8")
    result = runner.invoke(main, ["config", "--env-file", "selected.env"])
    assert result.exit_code == 0, result.output
    assert '"namespace": "selected"' in result.output
    result = runner.invoke(main, ["config", "--env-file", "missing.env"])
    assert result.exit_code == 2


def test_cli_options_and_benchmark_commands():
    # Dotenv loading has no opt-out; only production benchmark commands remain.
    for command in (main, *main.commands.values()):
        option_names = {parameter.name for parameter in command.params}
        assert "no_env_file" not in option_names
        assert "env_file" in option_names
    assert set(benchmark_main.commands) == {"hotpotqa", "locomo"}


def test_default_imports_without_optional_provider_packages():
    script = """
import importlib.abc
import sys

class BlockOptionalProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'langchain_huggingface', 'langchain_ollama'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockOptionalProviders())
from raganite.clients.llm import OpenAILLMClient, AzureOpenAILLMClient
from raganite.clients.llm.factory import create_chat_client, create_embedding_client
from raganite.clients.llm.usage import UsageReport, track_run
from raganite.config import ModelClientConfig
from raganite.graph import KnowledgeGraph
from raganite.retrieval import Retriever

for provider in ('huggingface', 'huggingface-endpoint', 'ollama'):
    try:
        create_chat_client(ModelClientConfig(provider=provider, model='test'))
    except ImportError as exc:
        assert 'optional extra' in str(exc)
    else:
        raise AssertionError('Missing extra should fail only when selected')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
