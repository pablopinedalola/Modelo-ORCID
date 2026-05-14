"""
retriever.py -- Hybrid Retriever (Vector + Graph).
"""

from typing import Any, Dict, List, Optional

from config import RAG_TOP_K
from src.rag.graph_retriever import GraphRetriever
from src.rag.reranker import HybridReranker
from src.rag.vector_store import VectorStore


class HybridRetriever:
    """Combina recuperación vectorial y basada en grafos."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_retriever = GraphRetriever()
        self.reranker = HybridReranker()

    def search(self, query: str, context_id: Optional[str] = None, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
        """Búsqueda híbrida.
        
        Args:
            query: La consulta del usuario.
            context_id: (Opcional) ID del investigador en contexto para boostear vecindad.
            top_k: Número de resultados.
            
        Returns:
            Lista de resultados reranqueados.
        """
        # 1. Recuperación Vectorial
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        
        # 2. Recuperación en Grafo (si hay contexto)
        graph_results = []
        if context_id:
            graph_results = self.graph_retriever.get_neighborhood(context_id, max_depth=1)
            
        # 3. Reranking
        final_results = self.reranker.rerank(vector_results, graph_results)
        
        return final_results[:top_k]
