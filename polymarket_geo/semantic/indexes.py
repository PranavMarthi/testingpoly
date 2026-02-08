from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polymarket_geo.semantic.embedder import LocalEmbeddingModel


@dataclass(frozen=True)
class IndexRecord:
    doc_id: str
    index_type: str
    place_id: str
    place_name: str
    granularity: str
    lat: float
    lon: float
    importance: float
    searchable_text: str


class LocalIndexes:
    def __init__(self, base_dir: Path | None = None, embed_dim: int = 384):
        self.base_dir = base_dir or Path(__file__).resolve().parents[2] / "data" / "semantic"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = LocalEmbeddingModel(dim=embed_dim)
        self.records: list[IndexRecord] = []
        self.matrix = np.zeros((0, embed_dim), dtype=np.float32)
        self._load_or_build_seed()

    def _load_or_build_seed(self) -> None:
        records_file = self.base_dir / "records.jsonl"
        vectors_file = self.base_dir / "vectors.npy"

        if records_file.exists() and vectors_file.exists():
            self.records = self._read_records(records_file)
            self.matrix = np.load(vectors_file)
            return

        seed_file = self.base_dir / "seed_records.jsonl"
        if not seed_file.exists():
            raise FileNotFoundError(
                f"Missing semantic seed dataset: {seed_file}. "
                "Provide local data and run scripts/build_indexes.py"
            )

        self.records = self._read_records(seed_file)
        self.matrix = self.embedder.embed_many([r.searchable_text for r in self.records])
        self._write_records(records_file, self.records)
        np.save(vectors_file, self.matrix)

    def _read_records(self, file_path: Path) -> list[IndexRecord]:
        out: list[IndexRecord] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                out.append(IndexRecord(**row))
        return out

    def _write_records(self, file_path: Path, records: list[IndexRecord]) -> None:
        with file_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.__dict__) + "\n")
