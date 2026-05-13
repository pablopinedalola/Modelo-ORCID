"""
Módulo RAG (Retrieval-Augmented Generation) para Modelo-ORCID.

Componentes:
    - EmbeddingPipeline: Genera y persiste embeddings de perfiles SNII.
    - FAISSStore: Índice vectorial FAISS con metadata y persistencia.
    - BasicRetriever: Búsqueda semántica top-k sobre perfiles.
"""

from .embedding_pipeline import EmbeddingPipeline
from .faiss_store import FAISSStore
from .basic_retriever import BasicRetriever
from .hybrid_retriever import HybridRetriever
from .graph_aware_retriever import GraphAwareRetriever

__all__ = ["EmbeddingPipeline", "FAISSStore", "BasicRetriever", "HybridRetriever", "GraphAwareRetriever"]
