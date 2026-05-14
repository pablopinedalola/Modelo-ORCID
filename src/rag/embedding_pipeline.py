"""
embedding_pipeline.py — Pipeline de generación de embeddings para perfiles SNII.

Transforma perfiles académicos en vectores semánticos densos usando
sentence-transformers (all-MiniLM-L6-v2). Soporta:
    - Generación batch eficiente
    - Caching en disco (pickle)
    - Múltiples campos: searchable_text, disciplina, área, institución
    - Persistencia local en data/vector_store/

El embedding principal se genera a partir del searchable_text, que ya
contiene toda la información relevante del perfil normalizada y sin acentos.

Examples:
    >>> from src.rag.embedding_pipeline import EmbeddingPipeline
    >>> pipeline = EmbeddingPipeline()
    >>> profiles = json.load(open("data/processed/snii_profiles.json"))
    >>> result = pipeline.generate_embeddings(profiles)
    >>> print(f"Generados {result['total']} embeddings de dim {result['dimension']}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import BASE_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Directorio de almacenamiento por defecto
DEFAULT_STORE_DIR = BASE_DIR / "data" / "vector_store"


class EmbeddingPipeline:
    """Pipeline de generación y persistencia de embeddings académicos.

    Genera embeddings densos para perfiles SNII usando sentence-transformers.
    Implementa caching en disco para evitar re-generación innecesaria.

    Attributes:
        model_name: Nombre del modelo sentence-transformers.
        store_dir: Directorio para persistir embeddings.
        batch_size: Tamaño del batch para encoding.
        _model: Modelo sentence-transformers (carga lazy).
        _cache: Cache en memoria {hash_texto: embedding}.

    Examples:
        >>> pipeline = EmbeddingPipeline()
        >>> result = pipeline.generate_embeddings(profiles)
        >>> pipeline.save(result)
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        store_dir: Optional[Path] = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.batch_size = batch_size
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._dimension: Optional[int] = None

        # Crear directorio si no existe
        self.store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        """Carga lazy del modelo sentence-transformers."""
        if self._model is None:
            logger.info(f"  Cargando modelo de embeddings: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            dim_fn = getattr(self._model, "get_embedding_dimension",
                             getattr(self._model, "get_sentence_embedding_dimension", None))
            self._dimension = dim_fn() if dim_fn else 384
            logger.info(f"  Modelo cargado: {self.model_name} (dim={self._dimension})")
        return self._model

    @property
    def dimension(self) -> int:
        """Dimensión de los embeddings del modelo."""
        if self._dimension is None:
            _ = self.model  # trigger load
        return self._dimension

    def generate_embeddings(
        self,
        profiles: list[dict],
        fields: Optional[list[str]] = None,
    ) -> dict:
        """Genera embeddings para todos los perfiles.

        Args:
            profiles: Lista de diccionarios de perfil (de snii_profiles.json).
            fields: Campos a embeber. Default: ["searchable_text"].
                    Opciones: searchable_text, disciplina, area_nombre, institucion.

        Returns:
            Diccionario con:
                - embeddings: np.ndarray shape (N, D)
                - profile_ids: lista de IDs correspondientes
                - metadata: lista de dicts con info de cada perfil
                - field: campo usado para embeddings
                - dimension: dimensión del vector
                - total: número de embeddings
                - model: nombre del modelo
        """
        if fields is None:
            fields = ["searchable_text"]

        primary_field = fields[0]
        logger.info(f"  Generando embeddings para campo: '{primary_field}'")
        logger.info(f"  Perfiles a procesar: {len(profiles)}")

        # Preparar textos y metadata
        texts = []
        profile_ids = []
        metadata_list = []
        skipped = 0

        for profile in profiles:
            text = self._extract_text(profile, primary_field)
            if not text or not text.strip():
                skipped += 1
                continue

            texts.append(text.strip())
            profile_ids.append(profile["id"])
            metadata_list.append(self._extract_metadata(profile))

        if skipped > 0:
            logger.warning(f"  ⚠️  {skipped} perfiles sin texto en '{primary_field}', omitidos")

        if not texts:
            logger.error("  ❌ No hay textos para generar embeddings")
            return {"embeddings": np.array([]), "profile_ids": [], "metadata": [],
                    "field": primary_field, "dimension": 0, "total": 0, "model": self.model_name}

        # Generar embeddings en batches
        embeddings = self._batch_encode(texts)

        logger.info(f"  ✅ {len(embeddings)} embeddings generados (dim={embeddings.shape[1]})")

        # Generar embeddings secundarios si se piden múltiples campos
        secondary_embeddings = {}
        for field in fields[1:]:
            sec_texts = []
            for profile in profiles:
                if profile["id"] in profile_ids:
                    t = self._extract_text(profile, field)
                    sec_texts.append(t if t else "")

            if sec_texts:
                sec_emb = self._batch_encode(sec_texts)
                secondary_embeddings[field] = sec_emb
                logger.info(f"  ✅ Embeddings secundarios '{field}': {sec_emb.shape}")

        return {
            "embeddings": embeddings,
            "profile_ids": profile_ids,
            "metadata": metadata_list,
            "field": primary_field,
            "dimension": int(embeddings.shape[1]),
            "total": len(embeddings),
            "model": self.model_name,
            "secondary_embeddings": secondary_embeddings,
        }

    def _extract_text(self, profile: dict, field: str) -> str:
        """Extrae el texto de un perfil para el campo solicitado.

        Args:
            profile: Diccionario del perfil.
            field: Campo a extraer.

        Returns:
            Texto del campo o string vacío.
        """
        if field == "searchable_text":
            return profile.get("searchable_text", "")
        elif field == "disciplina":
            return profile.get("disciplina", "")
        elif field == "area_nombre":
            return profile.get("area_nombre", "")
        elif field == "institucion":
            return profile.get("institucion", "")
        elif field == "combined":
            # Combinar varios campos para un embedding más rico
            parts = [
                profile.get("nombre_completo", ""),
                profile.get("disciplina", ""),
                profile.get("area_nombre", ""),
                profile.get("institucion", ""),
                profile.get("dependencia", ""),
            ]
            return " ".join(p for p in parts if p)
        else:
            return profile.get(field, "")

    @staticmethod
    def _extract_metadata(profile: dict) -> dict:
        """Extrae metadata compacta de un perfil para almacenar junto al embedding."""
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

    def _batch_encode(self, texts: list[str]) -> np.ndarray:
        """Codifica textos en batches para eficiencia de memoria.

        Args:
            texts: Lista de textos a codificar.

        Returns:
            np.ndarray de shape (N, D).
        """
        all_embeddings = []
        total = len(texts)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = texts[start:end]

            # Verificar cache
            uncached_texts = []
            uncached_indices = []
            batch_results = [None] * len(batch)

            for i, text in enumerate(batch):
                cache_key = self._text_hash(text)
                if cache_key in self._cache:
                    batch_results[i] = self._cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Codificar textos no cacheados
            if uncached_texts:
                encoded = self.model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )
                for idx, emb in zip(uncached_indices, encoded):
                    cache_key = self._text_hash(batch[idx])
                    self._cache[cache_key] = emb
                    batch_results[idx] = emb

            all_embeddings.extend(batch_results)

            if total > self.batch_size:
                progress = min(end, total)
                logger.info(f"    Batch {start//self.batch_size + 1}: {progress}/{total} textos")

        return np.array(all_embeddings, dtype=np.float32)

    @staticmethod
    def _text_hash(text: str) -> str:
        """Hash rápido para caching."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def encode_query(self, query: str) -> np.ndarray:
        """Codifica una consulta de búsqueda.

        Args:
            query: Texto de la consulta.

        Returns:
            Vector numpy de dimensión D.
        """
        embedding = self.model.encode(
            query.strip(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def save(self, result: dict, prefix: str = "snii") -> dict[str, Path]:
        """Persiste embeddings y metadata a disco.

        Args:
            result: Diccionario retornado por generate_embeddings().
            prefix: Prefijo para nombres de archivo.

        Returns:
            Dict con paths de los archivos guardados.
        """
        paths = {}

        # Embeddings como numpy
        emb_path = self.store_dir / f"{prefix}_embeddings.npy"
        np.save(emb_path, result["embeddings"])
        paths["embeddings"] = emb_path

        # Metadata como JSON
        meta_path = self.store_dir / f"{prefix}_metadata.json"
        meta = {
            "profile_ids": result["profile_ids"],
            "metadata": result["metadata"],
            "field": result["field"],
            "dimension": result["dimension"],
            "total": result["total"],
            "model": result["model"],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        paths["metadata"] = meta_path

        # Cache de embeddings (pickle)
        cache_path = self.store_dir / f"{prefix}_cache.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(self._cache, f)
        paths["cache"] = cache_path

        # Embeddings secundarios
        for field_name, sec_emb in result.get("secondary_embeddings", {}).items():
            sec_path = self.store_dir / f"{prefix}_{field_name}_embeddings.npy"
            np.save(sec_path, sec_emb)
            paths[f"secondary_{field_name}"] = sec_path

        logger.info(f"  💾 Embeddings guardados en {self.store_dir}/")
        for name, path in paths.items():
            size_kb = path.stat().st_size / 1024
            logger.info(f"     {name}: {path.name} ({size_kb:.1f} KB)")

        return paths

    def load_cache(self, prefix: str = "snii") -> bool:
        """Carga cache de embeddings desde disco.

        Args:
            prefix: Prefijo de archivos.

        Returns:
            True si se cargó exitosamente.
        """
        cache_path = self.store_dir / f"{prefix}_cache.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)
            logger.info(f"  Cache cargado: {len(self._cache)} embeddings")
            return True
        return False

    def stats(self) -> dict:
        """Estadísticas del pipeline."""
        return {
            "model": self.model_name,
            "dimension": self._dimension,
            "cached_embeddings": len(self._cache),
            "store_dir": str(self.store_dir),
        }
