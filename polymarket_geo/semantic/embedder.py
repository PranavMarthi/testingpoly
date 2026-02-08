from __future__ import annotations

import hashlib
import math
import re

import numpy as np


class LocalEmbeddingModel:
    """
    Lightweight offline semantic embedder.

    Uses hashed token + bigram projections into a dense vector with L2
    normalization. This is deterministic and requires no network/model downloads.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for tok in tokens:
            idx = self._hash(tok) % self.dim
            vec[idx] += 1.0

        for a, b in zip(tokens, tokens[1:]):
            idx = self._hash(f"{a}_{b}") % self.dim
            vec[idx] += 0.7

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return max(-1.0, min(1.0, float(np.dot(a, b) / (na * nb))))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        folded = text.lower()
        folded = folded.replace("u.s.a.", " usa ").replace("u.s.", " us ")
        folded = re.sub(r"[^a-z0-9\s]", " ", folded)
        folded = re.sub(r"\s+", " ", folded).strip()
        toks = [t for t in folded.split(" ") if t]
        norm: list[str] = []
        for t in toks:
            if t.endswith("ation"):
                norm.append(t[:-5])
            elif t.endswith("ing") and len(t) > 5:
                norm.append(t[:-3])
            elif t.endswith("s") and len(t) > 4:
                norm.append(t[:-1])
            norm.append(t)
        return norm

    @staticmethod
    def _hash(value: str) -> int:
        return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def score_to_unit(sim: float) -> float:
    # Keep only positive semantic alignment.
    return max(0.0, min(1.0, sim))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
