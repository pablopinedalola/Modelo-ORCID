"""
community_detection.py — Detección de comunidades científicas.

Aplica algoritmos de grafos (Louvain) sobre redes de colaboración.
"""

import logging
import networkx as nx

logger = logging.getLogger(__name__)

class CommunityDetector:
    """Detector de comunidades académicas."""

    def __init__(self, graph):
        self.graph = graph

    def detect_collaboration_communities(self) -> dict[int, list[str]]:
        """Usa Louvain para encontrar clusters de investigadores."""
        try:
            from networkx.algorithms.community import louvain_communities
        except ImportError:
            logger.warning("Louvain no disponible en esta versión de NetworkX.")
            return {}

        # 1. Extraer subgrafo undirected solo de candidatos/autores con coautoria
        # Para simplificar y usar la estructura actual, buscaremos autores conectados por el mismo paper
        author_graph = nx.Graph()
        
        for n, data in self.graph.nodes(data=True):
            if data.get("type") == "candidate":
                author_graph.add_node(n, label=data.get("label", n))
                
        # Conectar autores que comparten paper
        papers = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "paper"]
        for p in papers:
            authors = [pred for pred in self.graph.predecessors(p) if self.graph.nodes[pred].get("type") == "candidate"]
            import itertools
            for a1, a2 in itertools.combinations(authors, 2):
                if author_graph.has_edge(a1, a2):
                    author_graph[a1][a2]["weight"] += 1
                else:
                    author_graph.add_edge(a1, a2, weight=1)

        # Eliminar nodos aislados
        author_graph.remove_nodes_from(list(nx.isolates(author_graph)))
        
        if len(author_graph.nodes()) < 2:
            return {}

        communities = louvain_communities(author_graph, weight='weight')
        
        result = {}
        for i, comm in enumerate(communities):
            result[i] = [author_graph.nodes[n].get("label", n) for n in comm]
            
        return result
