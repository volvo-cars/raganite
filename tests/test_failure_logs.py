"""Caught failures stay in the run and are logged when verbose mode is on."""

import logging

from benchmark.cli import main as benchmark_main
from click.testing import CliRunner
from langchain_core.documents import Document

from raganite.backends.sqlite import SQLiteBackend
from raganite.cli import main as raganite_main
from raganite.config import RaganiteConfig, SQLiteConfig
from raganite.indexing.indexer import KnowledgeGraphIndexer
from raganite.log import configure_logging


class _Embeddings:
    async def aembed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_a_caught_indexing_failure_is_logged(caplog):
    backend = SQLiteBackend(SQLiteConfig(path=":memory:"), namespace="conv-26", dimensions=2)
    indexer = KnowledgeGraphIndexer(
        backend,
        object(),
        _Embeddings(),
        RaganiteConfig(namespace="conv-26", embedding_dimensions=2),
    )

    async def fail(*_args, **_kwargs):
        raise RuntimeError("bad extraction")

    indexer._build_graph = fail
    try:
        with caplog.at_level(logging.DEBUG, logger="raganite.indexing.indexer"):
            report = indexer.index_documents(
                [Document(page_content="hello", metadata={"path": "conv-26/session_011.txt"})],
                source="conv-26",
            )
        assert report.failed_files == [("conv-26/session_011.txt", "bad extraction")]
        assert "conv-26/session_011.txt" in caplog.text
        assert "bad extraction" in caplog.text
    finally:
        backend.close()


def test_vv_enables_debug_and_the_default_stays_quiet():
    configure_logging(2)
    logger = logging.getLogger("raganite.indexing.indexer")
    assert logger.isEnabledFor(logging.DEBUG)
    configure_logging(0)
    assert not logger.isEnabledFor(logging.DEBUG)
    assert logger.isEnabledFor(logging.WARNING)


def test_cli_help_documents_vv():
    benchmark = CliRunner().invoke(benchmark_main, ["locomo", "run", "--help"])
    raganite = CliRunner().invoke(raganite_main, ["index", "--help"])
    assert benchmark.exit_code == 0
    assert raganite.exit_code == 0
    assert "-v, --verbose" in benchmark.output
    assert "-v, --verbose" in raganite.output
