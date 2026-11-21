"""Download benchmark datasets when the local copies are missing."""

from pathlib import Path

import httpx

HOTPOT_RELEASE = "https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/5ec05b38deecc3318bb432c69865959c56058990"
HOTPOT_QUERIES_URL = f"{HOTPOT_RELEASE}/hotpotqa.json"
HOTPOT_CORPUS_URL = f"{HOTPOT_RELEASE}/hotpotqa_corpus.json"
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json"
)


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
