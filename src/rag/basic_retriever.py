"""
basic_retriever.py — Retriever semántico básico para perfiles SNII.

Proporciona una interfaz simple para buscar investigadores por
similitud semántica usando el índice FAISS y el embedding pipeline.

Funcionalidades:
    - Búsqueda semántica top-k
    - Score ranking por cosine similarity
    - Recuperación de perfiles completos
    - Fallback a búsqueda por substring si FAISS no disponible

NO implementa todavía:
    - Hybrid retrieval (BM25 + semántico)
    - Reranking complejo
    - Graph traversal
    - LLM-based reasoning

Examples:
    >>> retriever = BasicRetriever()
    >>> results = retriever.search("teoría de gráficas UNAM")
    >>> for r in results:
    ...     print(f"{r['nombre_completo']} ({r['score']:.3f})")
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Optional

from config import BASE_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class BasicRetriever:
    """Retriever semántico básico sobre perfiles SNII.

    Carga el índice FAISS y el pipeline de embeddings para
    permitir búsquedas semánticas. Incluye un fallback a
    búsqueda por substring para cuando FAISS no está disponible.

    Attributes:
        store_dir: Directorio del vector store.
        faiss_store: Índice FAISS cargado.
        embedding_pipeline: Pipeline para generar query embeddings.
        profiles: Perfiles completos para enriquecer resultados.
        _ready: Si el retriever está listo para búsquedas semánticas.

    Examples:
        >>> retriever = BasicRetriever()
        >>> retriever.load()
        >>> results = retriever.search("redes complejas")
    """

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else (BASE_DIR / "data" / "vector_store")
        self.faiss_store = None
        self.embedding_pipeline = None
        self.profiles: list[dict] = []
        self._profiles_by_id: dict[str, dict] = {}
        self._ready: bool = False

    def load(self) -> bool:
        """Carga todos los componentes necesarios.

        Returns:
            True si el retriever está listo para búsquedas semánticas.
        """
        # 1. Cargar perfiles procesados
        profiles_path = PROCESSED_DATA_DIR / "snii_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, "r", encoding="utf-8") as f:
                self.profiles = json.load(f)
            self._profiles_by_id = {p["id"]: p for p in self.profiles}
            logger.info(f"  📋 {len(self.profiles)} perfiles cargados")
        else:
            logger.warning(f"  ⚠️  No se encontraron perfiles en {profiles_path}")
            return False

        # 2. Cargar FAISS store
        try:
            from src.rag.faiss_store import FAISSStore
            self.faiss_store = FAISSStore(store_dir=self.store_dir)
            if not self.faiss_store.load():
                logger.warning("  ⚠️  Índice FAISS no disponible, usando fallback")
                self.faiss_store = None
        except Exception as e:
            logger.warning(f"  ⚠️  Error cargando FAISS: {e}")
            self.faiss_store = None

        # 3. Cargar embedding pipeline (solo si hay FAISS)
        if self.faiss_store:
            try:
                from src.rag.embedding_pipeline import EmbeddingPipeline
                self.embedding_pipeline = EmbeddingPipeline(store_dir=self.store_dir)
                self._ready = True
                logger.info("  ✅ Retriever semántico listo")
            except Exception as e:
                logger.warning(f"  ⚠️  Error cargando embedding pipeline: {e}")
                self._ready = False

        return self._ready

    @property
    def is_ready(self) -> bool:
        """Si el retriever está listo para búsquedas semánticas."""
        return self._ready

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        include_explanation: bool = False,
    ) -> list[dict]:
        """Busca investigadores por similitud semántica.

        Si FAISS está disponible, usa búsqueda semántica.
        Si no, usa fallback a búsqueda por substring.

        Args:
            query: Consulta de búsqueda en lenguaje natural.
            top_k: Número máximo de resultados.
            min_score: Score mínimo para incluir.
            include_explanation: Si incluir explicación del matching.

        Returns:
            Lista de dicts con info del perfil + score + rank.
        """
        if not query or not query.strip():
            return []

        query = query.strip()

        if self._ready and self.faiss_store and self.embedding_pipeline:
            return self._semantic_search(query, top_k, min_score, include_explanation)
        else:
            return self._fallback_search(query, top_k)

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        include_explanation: bool,
    ) -> list[dict]:
        """Búsqueda semántica vía FAISS.

        Args:
            query: Consulta.
            top_k: Top K resultados.
            min_score: Score mínimo.
            include_explanation: Incluir explicación.

        Returns:
            Lista de resultados con score.
        """
        # Generar embedding de la consulta
        query_vector = self.embedding_pipeline.encode_query(query)

        # Buscar en FAISS
        raw_results = self.faiss_store.search(
            query_vector, top_k=top_k, min_score=min_score,
        )

        # Enriquecer con perfil completo
        results = []
        for r in raw_results:
            profile_id = r["profile_id"]
            full_profile = self._profiles_by_id.get(profile_id, {})

            result = {
                # Campos del perfil
                "id": profile_id,
                "nombre_completo": r.get("nombre_completo", ""),
                "full_name": r.get("nombre_completo", ""),  # Alias para compatibilidad
                "institucion": r.get("institucion", ""),
                "institution": r.get("institucion", ""),  # Alias
                "area": r.get("area", ""),
                "area_nombre": r.get("area_nombre", ""),
                "disciplina": r.get("disciplina", ""),
                "discipline": r.get("disciplina", ""),  # Alias
                "nivel": r.get("nivel", ""),
                "nivel_nombre": r.get("nivel_nombre", ""),
                "dependencia": r.get("dependencia", ""),
                # Search metadata
                "score": r["score"],
                "rank": r["rank"],
                "search_method": "semantic",
                # Campos extra del perfil completo
                "nombre": full_profile.get("nombre", ""),
                "paterno": full_profile.get("paterno", ""),
                "materno": full_profile.get("materno", ""),
                "subdependencia": full_profile.get("subdependencia", ""),
                "confidence": r["score"],  # Para compat con templates existentes
            }

            if include_explanation:
                result["explanation"] = (
                    f"Similitud semántica: {r['score']:.3f} "
                    f"(consulta: '{query}')"
                )

            results.append(result)

        return results

    def _fallback_search(self, query: str, top_k: int) -> list[dict]:
        """Búsqueda por substring como fallback.

        Args:
            query: Consulta.
            top_k: Máximo de resultados.

        Returns:
            Lista de resultados.
        """
        query_normalized = self._normalize(query)
        results = []

        for profile in self.profiles:
            searchable = profile.get("searchable_text", "")
            nombre = profile.get("nombre_completo", "").lower()

            # Calcular score simple por coincidencia
            score = 0.0
            if query_normalized in searchable:
                score = 0.7
            elif query_normalized in nombre:
                score = 0.8
            else:
                # Buscar tokens individuales
                tokens = query_normalized.split()
                matches = sum(1 for t in tokens if t in searchable)
                if matches > 0:
                    score = 0.3 + (0.4 * matches / len(tokens))

            if score > 0:
                results.append({
                    "id": profile["id"],
                    "nombre_completo": profile.get("nombre_completo", ""),
                    "full_name": profile.get("nombre_completo", ""),
                    "institucion": profile.get("institucion", ""),
                    "institution": profile.get("institucion", ""),
                    "area": profile.get("area", ""),
                    "area_nombre": profile.get("area_nombre", ""),
                    "disciplina": profile.get("disciplina", ""),
                    "discipline": profile.get("disciplina", ""),
                    "nivel": profile.get("nivel", ""),
                    "nivel_nombre": profile.get("nivel_nombre", ""),
                    "dependencia": profile.get("dependencia", ""),
                    "score": score,
                    "rank": 0,
                    "search_method": "fallback_substring",
                    "confidence": score,
                })

        # Ordenar por score y asignar rank
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results[:top_k]):
            r["rank"] = i + 1

        return results[:top_k]

    @staticmethod
    def _normalize(text: str) -> str:
        """Normaliza texto para búsqueda (sin acentos, lowercase)."""
        text = text.lower().strip()
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def stats(self) -> dict:
        """Estadísticas del retriever."""
        return {
            "ready": self._ready,
            "total_profiles": len(self.profiles),
            "faiss_loaded": self.faiss_store is not None,
            "faiss_vectors": (
                self.faiss_store.index.ntotal
                if self.faiss_store and self.faiss_store.index
                else 0
            ),
            "search_method": "semantic" if self._ready else "fallback_substring",
        }
