from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from polymarket_geo.semantic.indexes import IndexRecord
from polymarket_geo.semantic.embedder import LocalEmbeddingModel


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build(seed_file: Path, out_dir: Path, dim: int = 384) -> None:
    raw = _read_jsonl(seed_file)
    records = [IndexRecord(**row) for row in raw]
    embedder = LocalEmbeddingModel(dim=dim)
    vectors = embedder.embed_many([r.searchable_text for r in records])

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "records.jsonl", [r.__dict__ for r in records])
    np.save(out_dir / "vectors.npy", vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local semantic retrieval indexes.")
    parser.add_argument("--seed", default="data/semantic/seed_records.jsonl")
    parser.add_argument("--out", default="data/semantic")
    parser.add_argument("--dim", default=384, type=int)
    args = parser.parse_args()

    build(Path(args.seed), Path(args.out), dim=args.dim)
    print(f"Built semantic indexes in {args.out}")


if __name__ == "__main__":
    main()
