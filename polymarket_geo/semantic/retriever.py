from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polymarket_geo.semantic.embedder import LocalEmbeddingModel, score_to_unit
from polymarket_geo.semantic.indexes import IndexRecord, LocalIndexes


@dataclass(frozen=True)
class RetrievalHit:
    field: str
    record: IndexRecord
    score: float


class Retriever:
    def __init__(self, indexes: LocalIndexes):
        self.indexes = indexes
        self.embedder = indexes.embedder

    def retrieve(self, field_texts: dict[str, str], top_n: int = 8) -> dict[str, list[RetrievalHit]]:
        out: dict[str, list[RetrievalHit]] = {}
        if self.indexes.matrix.shape[0] == 0:
            return {k: [] for k in field_texts}

        for field, text in field_texts.items():
            if not text.strip():
                out[field] = []
                continue

            qv = self.embedder.embed(text)
            sims = self.indexes.matrix @ qv
            idxs = np.argsort(-sims)[:top_n]
            hits = [
                RetrievalHit(field=field, record=self.indexes.records[int(i)], score=score_to_unit(float(sims[int(i)])))
                for i in idxs
            ]
            out[field] = [h for h in hits if h.score >= 0.05]
        return out
