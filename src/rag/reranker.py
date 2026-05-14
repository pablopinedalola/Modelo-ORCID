"""
reranker.py -- Reranking de resultados híbridos.
"""

from typing import Any, Dict, List


class HybridReranker:
    """Reranker simple para combinar resultados de FAISS y del Grafo."""

    @staticmethod
    def rerank(vector_results: List[Dict[str, Any]], graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fusiona y reordena resultados de vector store y grafo.
        
        Deduplica y prioriza nodos que aparecen en ambas fuentes.
        """
        scored_items = {}
        
        # Procesar vector results
        for item in vector_results:
            rid = item["metadata"].get("researcher_id")
            if not rid:
                continue
                
            if rid not in scored_items:
                scored_items[rid] = {
                    "id": rid,
                    "name": item["metadata"].get("name"),
                    "score": item.get("score", 0.0),
                    "sources": ["vector"],
                    "chunks": [item["text"]]
                }
            else:
                scored_items[rid]["score"] = max(scored_items[rid]["score"], item.get("score", 0.0))
                scored_items[rid]["chunks"].append(item["text"])

        # Procesar graph results (boosting)
        for item in graph_results:
            node_id = item.get("id", "").replace("snii:", "")
            
            if node_id in scored_items:
                # Boost si está en ambos lados
                scored_items[node_id]["score"] += 0.5
                scored_items[node_id]["sources"].append("graph")
            else:
                # Agregar solo del grafo
                scored_items[node_id] = {
                    "id": node_id,
                    "name": item.get("label", node_id),
                    "score": 0.3, # Score base para grafos
                    "sources": ["graph"],
                    "chunks": [f"Nodo relacionado en el grafo: {item.get('label')} ({item.get('type')})"]
                }
                
        # Ordenar por score descendente
        results_list = list(scored_items.values())
        results_list.sort(key=lambda x: x["score"], reverse=True)
        
        return results_list
