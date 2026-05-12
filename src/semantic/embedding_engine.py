"""
embedding_engine.py -- Motor de embeddings semanticos.

Genera embeddings vectoriales para textos academicos usando
sentence-transformers (all-MiniLM-L6-v2). Proporciona cosine
similarity para comparar representaciones semanticas.

El modelo se carga lazy (primera invocacion) para no penalizar
el tiempo de import.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Cache global del modelo (singleton)
_model = None
_model_name = None


def _get_model():
    """Carga el modelo de embeddings (lazy singleton)."""
    global _model, _model_name
    if _model is None or _model_name != EMBEDDING_MODEL:
        logger.info(f"  Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        _model_name = EMBEDDING_MODEL
        logger.info(f"  Modelo cargado: {EMBEDDING_MODEL} (dim={_model.get_sentence_embedding_dimension()})")
    return _model


class EmbeddingEngine:
    """Motor de embeddings semanticos con sentence-transformers.

    Genera vectores densos para textos academicos y calcula
    similitud coseno entre representaciones.

    Attributes:
        model_name: Nombre del modelo sentence-transformers.
        _cache: Cache de embeddings {text_hash: vector}.

    Examples:
        >>> engine = EmbeddingEngine()
        >>> v1 = engine.embed_text("quantum physics particle")
        >>> v2 = engine.embed_text("nuclear physics research")
        >>> sim = engine.cosine_similarity(v1, v2)
        >>> print(f"Similarity: {sim:.3f}")
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self):
        """Acceso lazy al modelo."""
        return _get_model()

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Genera embedding para un texto.

        Args:
            text: Texto a embeber.

        Returns:
            Vector numpy (384-dim para MiniLM), o None si vacio.
        """
        if not text or not text.strip():
            return None

        text = text.strip()[:512]  # Limitar longitud

        # Cache por texto
        cache_key = text[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.model.encode(text, convert_to_numpy=True)
        self._cache[cache_key] = embedding
        return embedding

    def embed_texts(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        """Genera embeddings para multiples textos (batch).

        Args:
            texts: Lista de textos.

        Returns:
            Lista de embeddings.
        """
        results = []
        to_encode = []
        to_encode_idx = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append(None)
                continue
            text = text.strip()[:512]
            cache_key = text[:100]
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                results.append(None)  # Placeholder
                to_encode.append(text)
                to_encode_idx.append(i)

        if to_encode:
            embeddings = self.model.encode(to_encode, convert_to_numpy=True)
            for idx, emb in zip(to_encode_idx, embeddings):
                cache_key = to_encode[to_encode_idx.index(idx)][:100]
                self._cache[cache_key] = emb
                results[idx] = emb

        return results

    @staticmethod
    def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Calcula similitud coseno entre dos vectores.

        Args:
            a: Primer vector.
            b: Segundo vector.

        Returns:
            Similitud en [-1.0, 1.0]. Retorna 0.0 si algun vector es None.
        """
        if a is None or b is None:
            return 0.0

        a = a.flatten()
        b = b.flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def embed_researcher(self, record) -> Optional[np.ndarray]:
        """Genera embedding semantico para un investigador SNII.

        Combina disciplina, area y nombre de institucion
        en un texto descriptivo para el embedding.

        Args:
            record: NormalizedRecord.

        Returns:
            Embedding del perfil del investigador.
        """
        parts = []

        if record.original.disciplina:
            parts.append(record.original.disciplina)
        if record.original.area:
            parts.append(record.original.area_label)
        if record.normalized_institution:
            parts.append(record.normalized_institution)
        if record.original.dependencia:
            parts.append(record.original.dependencia)

        text = " | ".join(parts) if parts else ""
        return self.embed_text(text) if text else None

    def embed_candidate(self, candidate) -> Optional[np.ndarray]:
        """Genera embedding semantico para un candidato.

        Combina conceptos/topics, afiliaciones y nombre.

        Args:
            candidate: Candidate object.

        Returns:
            Embedding del perfil del candidato.
        """
        parts = []

        # Conceptos/topics
        if candidate.concepts:
            parts.append(" ".join(candidate.concepts[:8]))

        # Afiliaciones
        if candidate.affiliations:
            parts.append(candidate.affiliations[0])

        # Si no hay conceptos, usar nombre
        if not parts:
            parts.append(candidate.display_name)

        text = " | ".join(parts) if parts else ""
        return self.embed_text(text) if text else None

    def cache_stats(self) -> dict:
        """Estadisticas del cache."""
        return {
            "cached_embeddings": len(self._cache),
            "model": self.model_name,
        }
