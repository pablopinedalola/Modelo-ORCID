"""
faiss_store.py — Índice FAISS con metadata y persistencia local.

Gestiona un índice FAISS (IndexFlatIP para cosine similarity) que
permite búsqueda semántica top-k sobre embeddings de perfiles SNII.

Funcionalidades:
    - Creación de índice desde embeddings numpy
    - Normalización L2 para cosine similarity via inner product
    - Búsqueda top-k con scores
    - Persistencia: guardar/cargar índice + metadata
    - Lookup de metadata por posición o profile_id

Examples:
    >>> store = FAISSStore()
    >>> store.build_index(embeddings, profile_ids, metadata)
    >>> results = store.search(query_vector, top_k=5)
    >>> for r in results:
    ...     print(f"{r['nombre_completo']} — score: {r['score']:.4f}")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config import BASE_DIR

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = BASE_DIR / "data" / "vector_store"


class FAISSStore:
    """Índice vectorial FAISS con metadata para búsqueda semántica.

    Usa IndexFlatIP (Inner Product) sobre vectores normalizados L2
    para obtener cosine similarity como score de búsqueda.

    Attributes:
        store_dir: Directorio de persistencia.
        index: Índice FAISS.
        profile_ids: Lista de IDs de perfil (posición → ID).
        metadata: Lista de dicts con metadata por perfil.
        dimension: Dimensión de los vectores.
        _id_to_idx: Mapeo inverso profile_id → posición.

    Examples:
        >>> store = FAISSStore()
        >>> store.build_index(embeddings, profile_ids, metadata)
        >>> store.save()
        >>> # Después...
        >>> store2 = FAISSStore()
        >>> store2.load()
        >>> results = store2.search(query_vec, top_k=5)
    """

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.index: Optional[faiss.Index] = None
        self.profile_ids: list[str] = []
        self.metadata: list[dict] = []
        self.dimension: int = 0
        self._id_to_idx: dict[str, int] = {}

        self.store_dir.mkdir(parents=True, exist_ok=True)

    def build_index(
        self,
        embeddings: np.ndarray,
        profile_ids: list[str],
        metadata: list[dict],
    ) -> None:
        """Construye el índice FAISS desde embeddings.

        Normaliza los vectores L2 y crea un IndexFlatIP para
        que inner product ≈ cosine similarity.

        Args:
            embeddings: Matriz (N, D) de embeddings float32.
            profile_ids: Lista de N IDs de perfil.
            metadata: Lista de N dicts con metadata.

        Raises:
            ValueError: Si las dimensiones no coinciden.
        """
        if len(embeddings) != len(profile_ids) or len(embeddings) != len(metadata):
            raise ValueError(
                f"Mismatch: embeddings={len(embeddings)}, "
                f"ids={len(profile_ids)}, metadata={len(metadata)}"
            )

        self.dimension = embeddings.shape[1]
        self.profile_ids = list(profile_ids)
        self.metadata = list(metadata)
        self._id_to_idx = {pid: i for i, pid in enumerate(self.profile_ids)}

        # Normalizar L2 para cosine similarity via inner product
        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)

        # Crear índice Inner Product
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info(
            f"  📊 Índice FAISS construido: "
            f"{self.index.ntotal} vectores, dim={self.dimension}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Busca los top-k perfiles más similares a la consulta.

        Args:
            query_vector: Vector de consulta (D,) o (1, D).
            top_k: Número máximo de resultados.
            min_score: Score mínimo para incluir en resultados.

        Returns:
            Lista de dicts con metadata + score, ordenados por score desc.
            Cada dict incluye: score, rank, y todos los campos de metadata.

        Raises:
            RuntimeError: Si el índice no está construido.
        """
        if self.index is None:
            raise RuntimeError("Índice no construido. Usa build_index() o load() primero.")

        # Asegurar shape (1, D)
        query = query_vector.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(query)

        # Buscar
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS retorna -1 para slots vacíos
                continue
            if score < min_score:
                continue

            result = {
                "rank": rank + 1,
                "score": float(score),
                "profile_id": self.profile_ids[idx],
                **self.metadata[idx],
            }
            results.append(result)

        return results

    def search_by_text(
        self,
        query_text: str,
        encoder,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Busca por texto usando un encoder para generar el vector.

        Args:
            query_text: Texto de búsqueda.
            encoder: Objeto con método encode_query(text) -> np.ndarray.
            top_k: Número máximo de resultados.
            min_score: Score mínimo.

        Returns:
            Lista de resultados ordenados por score.
        """
        query_vector = encoder.encode_query(query_text)
        return self.search(query_vector, top_k=top_k, min_score=min_score)

    def get_by_id(self, profile_id: str) -> Optional[dict]:
        """Recupera metadata de un perfil por su ID.

        Args:
            profile_id: ID del perfil.

        Returns:
            Dict con metadata o None si no existe.
        """
        idx = self._id_to_idx.get(profile_id)
        if idx is not None:
            return self.metadata[idx]
        return None

    def get_vector(self, profile_id: str) -> Optional[np.ndarray]:
        """Recupera el vector de un perfil por su ID.

        Args:
            profile_id: ID del perfil.

        Returns:
            Vector numpy o None si no existe.
        """
        idx = self._id_to_idx.get(profile_id)
        if idx is not None and self.index is not None:
            return self.index.reconstruct(idx)
        return None

    def save(self, prefix: str = "snii") -> dict[str, Path]:
        """Persiste índice FAISS y metadata a disco.

        Args:
            prefix: Prefijo para los archivos.

        Returns:
            Dict con paths de archivos guardados.
        """
        if self.index is None:
            raise RuntimeError("No hay índice para guardar.")

        paths = {}

        # Índice FAISS
        index_path = self.store_dir / f"{prefix}_faiss.index"
        faiss.write_index(self.index, str(index_path))
        paths["index"] = index_path

        # Metadata + mappings
        store_meta = {
            "profile_ids": self.profile_ids,
            "metadata": self.metadata,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
        }
        meta_path = self.store_dir / f"{prefix}_faiss_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(store_meta, f, ensure_ascii=False, indent=2)
        paths["metadata"] = meta_path

        logger.info(f"  💾 Índice FAISS guardado:")
        for name, path in paths.items():
            size_kb = path.stat().st_size / 1024
            logger.info(f"     {name}: {path.name} ({size_kb:.1f} KB)")

        return paths

    def load(self, prefix: str = "snii") -> bool:
        """Carga índice FAISS y metadata desde disco.

        Args:
            prefix: Prefijo de los archivos.

        Returns:
            True si se cargó exitosamente, False si no existen los archivos.
        """
        index_path = self.store_dir / f"{prefix}_faiss.index"
        meta_path = self.store_dir / f"{prefix}_faiss_meta.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(f"  ⚠️  Archivos de índice no encontrados en {self.store_dir}")
            return False

        # Cargar índice
        self.index = faiss.read_index(str(index_path))

        # Cargar metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            store_meta = json.load(f)

        self.profile_ids = store_meta["profile_ids"]
        self.metadata = store_meta["metadata"]
        self.dimension = store_meta["dimension"]
        self._id_to_idx = {pid: i for i, pid in enumerate(self.profile_ids)}

        logger.info(
            f"  ✅ Índice FAISS cargado: "
            f"{self.index.ntotal} vectores, dim={self.dimension}"
        )
        return True

    def stats(self) -> dict:
        """Estadísticas del índice."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "total_profiles": len(self.profile_ids),
            "store_dir": str(self.store_dir),
            "index_loaded": self.index is not None,
        }

    def integrity_check(self) -> dict:
        """Verifica integridad del índice.

        Returns:
            Dict con resultados de la verificación.
        """
        checks = {
            "index_exists": self.index is not None,
            "vectors_match_ids": False,
            "vectors_match_metadata": False,
            "dimension_consistent": False,
            "all_passed": False,
        }

        if self.index is None:
            return checks

        checks["vectors_match_ids"] = self.index.ntotal == len(self.profile_ids)
        checks["vectors_match_metadata"] = self.index.ntotal == len(self.metadata)
        checks["dimension_consistent"] = self.index.d == self.dimension
        checks["all_passed"] = all([
            checks["index_exists"],
            checks["vectors_match_ids"],
            checks["vectors_match_metadata"],
            checks["dimension_consistent"],
        ])

        return checks
