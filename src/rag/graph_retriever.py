"""
graph_retriever.py -- Graph-Aware Retrieval.

Navega el Knowledge Graph para recuperar vecinos, coautores e instituciones.
"""

import json
import logging
from typing import Any, Dict, List, Set

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)

class GraphRetriever:
    """Recuperador basado en navegación del grafo de conocimiento."""

    def __init__(self):
        self.graph_path = OUTPUT_DIR / "knowledge_graph.json"
        self._graph_cache = None

    def _load_graph(self) -> Dict[str, Any]:
        if self._graph_cache:
            return self._graph_cache
        
        if self.graph_path.exists():
            with open(self.graph_path, "r", encoding="utf-8") as f:
                self._graph_cache = json.load(f)
        else:
            self._graph_cache = {"nodes": [], "edges": []}
            
        return self._graph_cache

    def get_neighborhood(self, node_id: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """Recupera la vecindad N(v) de un nodo en el grafo.
        
        Args:
            node_id: ID del nodo origen (ej. 'snii:1234' o '1234')
            max_depth: Profundidad de búsqueda (1 = vecinos directos)
            
        Returns:
            Lista de nodos vecinos enriquecidos.
        """
        graph = self._load_graph()
        
        # Normalizar ID si es necesario (el grafo guarda snii:XXX)
        if not node_id.startswith("snii:") and any(n["id"] == f"snii:{node_id}" for n in graph["nodes"]):
            node_id = f"snii:{node_id}"

        visited = {node_id}
        current_level = {node_id}
        
        for _ in range(max_depth):
            next_level = set()
            for edge in graph.get("edges", []):
                src = edge["source"]
                tgt = edge["target"]
                
                if src in current_level and tgt not in visited:
                    next_level.add(tgt)
                    visited.add(tgt)
                elif tgt in current_level and src not in visited:
                    next_level.add(src)
                    visited.add(src)
            current_level = next_level

        # Filtrar el origen de los resultados
        visited.remove(node_id) if node_id in visited else None
        
        neighbors = [n for n in graph.get("nodes", []) if n["id"] in visited]
        return neighbors

    def get_coauthors(self, node_id: str) -> List[Dict[str, Any]]:
        """Recupera específicamente los coautores."""
        neighbors = self.get_neighborhood(node_id, max_depth=1)
        return [n for n in neighbors if n.get("type") in ("openalex_author", "coauthor")]
