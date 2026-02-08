from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from polymarket_geo.semantic.composer import ComposedText
from polymarket_geo.semantic.embedder import LocalEmbeddingModel


@dataclass(frozen=True)
class EventTypePrototype:
    label: str
    text: str


class EventTypeClassifier:
    """Embedding-nearest classifier over local textual prototypes."""

    def __init__(self, embedder: LocalEmbeddingModel | None = None):
        self.embedder = embedder or LocalEmbeddingModel()
        self.prototypes = self._load_prototypes()
        self.proto_vecs = self.embedder.embed_many([p.text for p in self.prototypes])

    @staticmethod
    def _load_prototypes() -> list[EventTypePrototype]:
        file_path = Path(__file__).resolve().parents[2] / "data" / "semantic" / "event_type_prototypes.jsonl"
        if not file_path.exists():
            return []
        rows: list[EventTypePrototype] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(EventTypePrototype(label=row["label"], text=row["text"]))
        return rows

    def predict(self, composed: ComposedText) -> str:
        qv = self.embedder.embed(composed.combined_text)
        if self.proto_vecs.shape[0] == 0:
            return "unknown"
        sims = self.proto_vecs @ qv
        idx = int(sims.argmax())
        best = float(sims[idx])
        if best < 0.22:
            return "unknown"
        return self.prototypes[idx].label
