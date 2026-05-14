"""
vector_store.py -- FAISS Vector Store para persistencia y retrieval semántico.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Intentar importar faiss, manejar error si no está
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.warning("faiss-cpu no está instalado. VectorStore no funcionará.")

from config import VECTOR_STORE_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper sobre FAISS para indexación académica."""
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, index_dir: Path = VECTOR_STORE_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.metadata_path = self.index_dir / "metadata.json"
        
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.EMBEDDING_MODEL)
        self.dimension = 384  # Para all-MiniLM-L6-v2
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        
        if HAS_FAISS:
            self._load_or_create_index()

    def _load_or_create_index(self):
        if self.index_path.exists() and self.metadata_path.exists():
            logger.info(f"Cargando índice FAISS desde {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            logger.info("Creando nuevo índice FAISS")
            self.index = faiss.IndexFlatIP(self.dimension) # Inner Product (~Cosine)
            self.metadata = []

    def save(self):
        """Persiste el índice en disco."""
        if not HAS_FAISS or self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Índice guardado ({len(self.metadata)} vectores)")

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Añade chunks semánticos al índice."""
        if not HAS_FAISS or not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        valid_embs = []
        for i, emb in enumerate(embeddings):
            if emb is not None:
                # Normalizar para usar Inner Product como Cosine Similarity
                faiss.normalize_L2(emb.reshape(1, -1))
                valid_embs.append(emb)
                self.metadata.append(chunks[i])

        if valid_embs:
            emb_matrix = np.vstack(valid_embs).astype("float32")
            self.index.add(emb_matrix)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Busca los chunks más relevantes para una query."""
        if not HAS_FAISS or self.index is None or self.index.ntotal == 0:
            return []

        query_emb = self.model.encode(query, convert_to_numpy=True)
        if query_emb is None:
            return []

        query_emb = query_emb.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_emb)

        # Buscar
        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(score)
                results.append(item)

        return results
