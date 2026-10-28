from pathlib import Path

from benchmark.cli import main
from benchmark.fetch import ensure_download
from benchmark.hotpotqa_flow import corpus_documents
from benchmark.locomo import selected_samples, session_documents
from benchmark.store import maybe_index, store_config
from click.testing import CliRunner


def test_download_skips_a_file_that_already_has_bytes(tmp_path: Path):
    target = tmp_path / "hotpotqa.json"
    target.write_text("{}", encoding="utf-8")
    calls = []

    def get(*_args, **_kwargs):
        calls.append(True)

    assert ensure_download(target, "https://example.invalid/hotpotqa.json", get=get) == target
    assert calls == []


def test_corpus_documents_keep_the_passage_filename():
    documents, titles = corpus_documents(
        [{"idx": 7, "title": "Ada Lovelace", "text": "Mathematician."}]
    )
    assert documents[0].metadata["path"] == "doc_00007.txt"
    assert titles["doc_00007.txt"] == "Ada Lovelace"
    assert documents[0].page_content == "Mathematician."


def test_sqlite_and_neo4j_configs():
    sqlite = store_config(
        "sqlite",
        sqlite_path=Path("benchmark/data/hotpotqa.sqlite3"),
        neo4j_uri="",
        username="",
        password="",
        database="",
        dimensions=1536,
    )
    neo4j = store_config(
        "neo4j",
        sqlite_path=Path("unused.sqlite3"),
        neo4j_uri="bolt://localhost:7687",
        username="neo4j",
        password="raganite-local",
        database="neo4j",
        dimensions=3072,
    )
    assert sqlite.backend.kind == "sqlite"
    assert sqlite.backend.read_only is False
    assert neo4j.backend.kind == "neo4j"
    assert neo4j.backend.layout == "owned"
    assert neo4j.backend.read_only is False
    assert neo4j.embedding_dimensions == 3072


def test_skip_index_does_not_call_the_indexer():
    class Indexer:
        def __init__(self):
            self.calls = 0

        def index_documents(self, documents, **_kwargs):
            self.calls += 1
            return documents

    indexer = Indexer()
    assert maybe_index(indexer, ["passage"], source="hotpotqa", skip_index=True) is None
    assert indexer.calls == 0
    assert maybe_index(indexer, ["passage"], source="hotpotqa", skip_index=False) == ["passage"]
    assert indexer.calls == 1


def test_locomo_session_keeps_dialogue_ids_and_stays_in_one_conversation():
    sample = {
        "sample_id": "conv-26",
        "conversation": {
            "speaker_a": "A",
            "session_1": [{"dia_id": "D1:3", "speaker": "A", "text": "I moved."}],
            "session_1_date_time": "1:00 pm on 8 May, 2023",
        },
        "qa": [{"question": "When?", "answer": "May", "category": 2, "evidence": ["D1:3"]}],
    }
    documents = session_documents(sample)
    assert documents[0].metadata["path"] == "conv-26/session_001.txt"
    assert documents[0].metadata["order"] == 1
    assert documents[0].metadata["time"] == "1:00 pm on 8 May, 2023"
    assert "[D1:3] A (1:00 pm on 8 May, 2023): I moved." in documents[0].page_content
    chosen = selected_samples([sample, {"sample_id": "conv-30", "qa": [{}]}], 1)
    assert [item[0]["sample_id"] for item in chosen] == ["conv-26"]


def test_help_lists_index_backends_and_skip():
    result = CliRunner().invoke(main, ["hotpotqa", "--help"])
    assert result.exit_code == 0
    assert "--backend" in result.output
    assert "--skip-index" in result.output
    assert "sqlite" in result.output
    assert "neo4j" in result.output
    locomo = CliRunner().invoke(main, ["locomo", "run", "--help"])
    assert locomo.exit_code == 0
    assert "--skip-index" in locomo.output
    assert "--schema-mode" in locomo.output
    assert "free" in locomo.output


def test_locomo_indexing_uses_the_requested_schema(tmp_path: Path, monkeypatch):
    seen = {}

    class Graph:
        def __init__(self, **kwargs):
            seen["schema_mode"] = kwargs["config"].indexing.schema_mode
            self.indexer = None
            self.backend = None

        def close(self):
            return None

    class Retriever:
        def __init__(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    def close_coroutine(coroutine):
        coroutine.close()

    monkeypatch.setattr("raganite.graph.KnowledgeGraph", Graph)
    monkeypatch.setattr("raganite.retrieval.Retriever", Retriever)
    monkeypatch.setattr(
        "benchmark.locomo_flow._clients", lambda _dimensions, **_kwargs: (None, None, None, [])
    )
    monkeypatch.setattr("benchmark.locomo_flow.asyncio.run", close_coroutine)
    sample = {
        "sample_id": "conv-26",
        "conversation": {
            "session_1": [{"dia_id": "D1:1", "speaker": "Caroline", "text": "I moved."}],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
        },
    }
    options = {
        "concurrency": 1,
        "top_k": 20,
        "backend": "sqlite",
        "skip_index": True,
        "sqlite_path": tmp_path / "locomo.sqlite3",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "raganite-local",
        "neo4j_database": "neo4j",
        "neo4j_layout": "owned",
        "output": tmp_path / "locomo.jsonl",
        "schema_mode": "free",
    }
    from benchmark.locomo_flow import _run_locomo_conversations

    _run_locomo_conversations(options, [(sample, [])], 1536)
    assert seen["schema_mode"] == "free"
    options["schema_mode"] = "fixed"
    _run_locomo_conversations(options, [(sample, [])], 1536)
    assert seen["schema_mode"] == "fixed"
    options.pop("schema_mode")
    _run_locomo_conversations(options, [(sample, [])], 1536)
    assert seen["schema_mode"] == "free"
