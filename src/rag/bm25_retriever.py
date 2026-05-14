"""
bm25_retriever.py — Retriever léxico BM25 para perfiles SNII.

Implementa búsqueda léxica usando BM25 (Okapi) sobre los perfiles
SNII procesados. Complementa al retriever semántico (FAISS) para
el motor de retrieval híbrido.

Funcionalidades:
    - Tokenización académica (sin acentos, stopwords, bigramas)
    - Indexación de searchable_text + campos individuales
    - Búsqueda léxica top-k con score BM25
    - Persistencia local (pickle)
    - Explicación de matches (qué tokens coincidieron)

Examples:
    >>> retriever = BM25Retriever()
    >>> retriever.build_index(profiles)
    >>> results = retriever.search("física UNAM partículas", top_k=5)
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from config import BASE_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = BASE_DIR / "data" / "vector_store"

# ─── Stopwords académicas (español/inglés comunes en perfiles SNII) ───
STOPWORDS = frozenset({
    # Español
    "de", "del", "la", "las", "los", "el", "en", "y", "a", "e", "o", "u",
    "un", "una", "unos", "unas", "por", "para", "con", "sin", "sobre",
    "que", "es", "se", "su", "al", "lo", "como", "mas", "pero",
    # Inglés
    "the", "of", "and", "in", "for", "to", "a", "an", "is", "on", "at",
    "with", "from", "by",
    # Institucionales
    "universidad", "instituto", "centro", "facultad", "escuela",
    "departamento", "nacional", "autonoma",
})


class BM25Retriever:
    """Retriever léxico BM25 para perfiles académicos.

    Usa BM25Okapi de rank-bm25 para búsqueda léxica sobre tokens
    de los perfiles SNII. Complementa la búsqueda semántica con
    matching exacto de keywords.

    Attributes:
        store_dir: Directorio de persistencia.
        bm25: Índice BM25.
        profile_ids: IDs de perfil indexados.
        metadata: Metadata de cada perfil.
        corpus_tokens: Tokens del corpus (para explicación).
        profiles: Perfiles completos.

    Examples:
        >>> bm25 = BM25Retriever()
        >>> bm25.build_index(profiles)
        >>> results = bm25.search("óptica cuántica", top_k=5)
    """

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.bm25: Optional[BM25Okapi] = None
        self.profile_ids: list[str] = []
        self.metadata: list[dict] = []
        self.corpus_tokens: list[list[str]] = []
        self.profiles: list[dict] = []
        self._id_to_idx: dict[str, int] = {}

        self.store_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, profiles: list[dict]) -> None:
        """Construye el índice BM25 desde perfiles procesados.

        Tokeniza el searchable_text de cada perfil y construye un
        corpus BM25. También indexa campos individuales para
        búsquedas específicas.

        Args:
            profiles: Lista de dicts de perfil (de snii_profiles.json).
        """
        self.profiles = profiles
        self.profile_ids = []
        self.metadata = []
        self.corpus_tokens = []

        for profile in profiles:
            # Construir texto para indexar (enriquecido)
            text = self._build_index_text(profile)
            tokens = self.tokenize(text)

            if not tokens:
                continue

            self.profile_ids.append(profile["id"])
            self.metadata.append(self._extract_metadata(profile))
            self.corpus_tokens.append(tokens)

        self._id_to_idx = {pid: i for i, pid in enumerate(self.profile_ids)}

        # Construir BM25
        self.bm25 = BM25Okapi(self.corpus_tokens)

        logger.info(
            f"  📖 Índice BM25 construido: "
            f"{len(self.profile_ids)} documentos, "
            f"vocab≈{len(set(t for doc in self.corpus_tokens for t in doc))} tokens"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Busca perfiles por coincidencia léxica BM25.

        Args:
            query: Consulta de búsqueda.
            top_k: Número máximo de resultados.
            min_score: Score BM25 mínimo.

        Returns:
            Lista de dicts con metadata + score, ordenados por score desc.
        """
        if self.bm25 is None:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Obtener scores BM25
        scores = self.bm25.get_scores(query_tokens)

        # Ranking
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score < min_score:
                continue

            result = {
                "rank": rank + 1,
                "score": score,
                "profile_id": self.profile_ids[idx],
                **self.metadata[idx],
            }
            results.append(result)

        return results

    def search_with_explanation(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Busca con explicación de qué tokens coincidieron.

        Args:
            query: Consulta.
            top_k: Máximo resultados.

        Returns:
            Lista de resultados con campo 'matched_tokens'.
        """
        results = self.search(query, top_k=top_k)
        query_tokens = set(self.tokenize(query))

        for r in results:
            idx = self._id_to_idx.get(r["profile_id"])
            if idx is not None:
                doc_tokens = set(self.corpus_tokens[idx])
                matched = query_tokens & doc_tokens
                r["matched_tokens"] = sorted(matched)
                r["token_overlap"] = len(matched) / len(query_tokens) if query_tokens else 0
            else:
                r["matched_tokens"] = []
                r["token_overlap"] = 0.0

        return results

    @staticmethod
    def _build_index_text(profile: dict) -> str:
        """Construye texto enriquecido para indexar.

        Duplica campos importantes para darles más peso en BM25.
        """
        aliases = " ".join(profile.get("aliases", []))
        publications = " ".join(profile.get("publications", []))
        topics = " ".join(profile.get("topics", []))
        
        parts = [
            # searchable_text ya tiene todo normalizado
            profile.get("searchable_text", ""),
            # Aliases para matching exacto
            aliases,
            # Publicaciones y Topics (NUEVO)
            publications,
            topics,
            # Duplicar campos clave para más peso
            profile.get("nombre_completo", ""),
            profile.get("disciplina", ""),
            profile.get("disciplina", ""),  # doble peso disciplina
            profile.get("institucion", ""),
            profile.get("area_nombre", ""),
            profile.get("dependencia", ""),
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _extract_metadata(profile: dict) -> dict:
        """Extrae metadata compacta."""
        return {
            "id": profile.get("id", ""),
            "nombre_completo": profile.get("nombre_completo", ""),
            "normalized_name": profile.get("normalized_name", ""),
            "aliases": profile.get("aliases", []),
            "institucion": profile.get("institucion", ""),
            "area": profile.get("area", ""),
            "area_nombre": profile.get("area_nombre", ""),
            "disciplina": profile.get("disciplina", ""),
            "nivel": profile.get("nivel", ""),
            "nivel_nombre": profile.get("nivel_nombre", ""),
            "dependencia": profile.get("dependencia", ""),
        }

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Tokeniza texto para BM25 con normalización académica.

        - Lowercase
        - Remove acentos/diacríticos
        - Split en tokens alfanuméricos
        - Remove stopwords
        - Mantiene tokens de 2+ caracteres

        Args:
            text: Texto a tokenizar.

        Returns:
            Lista de tokens.
        """
        if not text:
            return []

        # Lowercase + remove acentos
        text = text.lower().strip()
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in nfkd if not unicodedata.combining(c))

        # Tokenizar (solo alfanuméricos)
        tokens = re.findall(r"[a-z0-9]+", text)

        # Filtrar stopwords y tokens cortos
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]

        return tokens

    def save(self, prefix: str = "snii") -> Path:
        """Persiste índice BM25 a disco.

        Args:
            prefix: Prefijo para el archivo.

        Returns:
            Path del archivo guardado.
        """
        data = {
            "profile_ids": self.profile_ids,
            "metadata": self.metadata,
            "corpus_tokens": self.corpus_tokens,
        }
        path = self.store_dir / f"{prefix}_bm25.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)

        size_kb = path.stat().st_size / 1024
        logger.info(f"  💾 Índice BM25 guardado: {path.name} ({size_kb:.1f} KB)")
        return path

    def load(self, prefix: str = "snii") -> bool:
        """Carga índice BM25 desde disco.

        Args:
            prefix: Prefijo del archivo.

        Returns:
            True si se cargó exitosamente.
        """
        path = self.store_dir / f"{prefix}_bm25.pkl"
        if not path.exists():
            logger.warning(f"  ⚠️  Índice BM25 no encontrado: {path}")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.profile_ids = data["profile_ids"]
        self.metadata = data["metadata"]
        self.corpus_tokens = data["corpus_tokens"]
        self._id_to_idx = {pid: i for i, pid in enumerate(self.profile_ids)}

        # Reconstruir BM25
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # Cargar perfiles completos
        profiles_path = PROCESSED_DATA_DIR / "snii_profiles.json"
        if profiles_path.exists():
            with open(profiles_path) as f:
                self.profiles = json.load(f)

        logger.info(f"  ✅ Índice BM25 cargado: {len(self.profile_ids)} documentos")
        return True

    def stats(self) -> dict:
        """Estadísticas del índice."""
        total_tokens = sum(len(doc) for doc in self.corpus_tokens)
        unique_tokens = len(set(t for doc in self.corpus_tokens for t in doc))
        return {
            "total_documents": len(self.profile_ids),
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_tokens_per_doc": total_tokens / len(self.profile_ids) if self.profile_ids else 0,
            "index_loaded": self.bm25 is not None,
        }
