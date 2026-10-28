"""Result persistence and corpus-checked passage mapping for benchmarks."""

import json
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError("Duplicate question ID in results")
                rows[row["id"]] = row
    return rows


def save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def passages_from_rows(rows: list[dict], title_map: dict[str, str], top_k: int) -> list[dict]:
    passages, seen = [], set()
    for row in rows:
        source = row.get("source") or ""
        filename = Path(source).name
        if not filename or filename in seen:
            continue
        if filename not in title_map:
            raise ValueError("Retrieved source is not in the expected corpus")
        seen.add(filename)
        passages.append(
            {
                "source": source,
                "title": title_map[filename],
                "text": row["text"],
                "score": row.get("rrf", row["score"]),
                "node_id": row["node_id"],
            }
        )
    return passages[:top_k]
