"""
scientific_metrics.py — Métricas de impacto e influencia.
"""

import logging
import networkx as nx

from src.models.schemas import NodeType

logger = logging.getLogger(__name__)

class ScientificMetrics:
    """Calcula métricas científicas complejas sobre el grafo."""

    def __init__(self, graph):
        self.graph = graph

    def get_institution_productivity(self) -> dict[str, int]:
        """Calcula productividad de instituciones contando papers de sus autores."""
        prod = {}
        for inst, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.INSTITUTION.value:
                # Contar papers de investigadores afiliados a esta institucion
                paper_count = 0
                for pred in self.graph.predecessors(inst):
                    # pred es un snii o cand.
                    for p_succ in self.graph.successors(pred):
                        if self.graph.nodes[p_succ].get("type") == NodeType.PAPER.value:
                            paper_count += 1
                if paper_count > 0:
                    prod[data.get("label", inst)] = paper_count
                    
        return dict(sorted(prod.items(), key=lambda item: item[1], reverse=True))

    def get_author_centrality(self, top_k=10):
        """Calcula PageRank sobre la red de autores (hubs de conocimiento)."""
        author_graph = nx.DiGraph()
        
        for u, v, data in self.graph.edges(data=True):
            type_u = self.graph.nodes[u].get("type")
            type_v = self.graph.nodes[v].get("type")
            
            # Aproximar influencia: si A cita un paper de B -> A es influenciado por B
            if data.get("type") == "cites":
                # u = paper 1, v = paper 2
                authors_u = [a for a in self.graph.predecessors(u) if self.graph.nodes[a].get("type") == "candidate"]
                authors_v = [a for a in self.graph.predecessors(v) if self.graph.nodes[a].get("type") == "candidate"]
                
                for au in authors_u:
                    for av in authors_v:
                        if au != av:
                            author_graph.add_edge(au, av)
                            
        if not author_graph.nodes():
            return []
            
        pr = nx.pagerank(author_graph)
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            {
                "author": self.graph.nodes[n].get("label", n),
                "centrality": score
            }
            for n, score in sorted_pr
        ]
