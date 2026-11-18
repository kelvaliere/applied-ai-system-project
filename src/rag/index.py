"""Retriever: albums as IDF-weighted genre vectors, cosine similarity (numpy only, ~100 docs)."""
from typing import Dict, List

import numpy as np

from .crosscheck import AlbumRecord


class TagIndex:
    def __init__(self, records: List[AlbumRecord]):
        self.records = records
        self.vocab = {t: i for i, t in enumerate(sorted({t for r in records for t in r.genres}))}
        weights = np.zeros((len(records), len(self.vocab)))
        for row, rec in enumerate(records):
            for tag, w in rec.genres.items():
                weights[row, self.vocab[tag]] = w
        df = (weights > 0).sum(axis=0)
        self.idf = np.log((1 + len(records)) / (1 + df)) + 1.0     # common tags like "rock" count less
        self.matrix = self._normalize(weights * self.idf)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)

    def vector(self, tags: Dict[str, float]) -> np.ndarray:
        """Unit vector for a {tag: weight} dict; tags unseen in the corpus are ignored (zero vector if none known)."""
        v = np.zeros(len(self.vocab))
        for tag, w in tags.items():
            if tag in self.vocab:
                v[self.vocab[tag]] = w
        return self._normalize(v * self.idf)

    def similarity(self, vec: np.ndarray) -> np.ndarray:
        return self.matrix @ vec
