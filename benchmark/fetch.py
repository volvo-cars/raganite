"""Download benchmark datasets when the local copies are missing."""

from pathlib import Path

import httpx

HOTPOT_QUERIES_URL = "https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/main/hotpotqa.json"
HOTPOT_CORPUS_URL = (
    "https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/main/hotpotqa_corpus.json"
)
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def ensure_download(path: Path, url: str, *, get=None) -> Path:
    """Return ``path``, downloading ``url`` when the file is missing or empty."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = get or httpx.get
    response = request(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path
