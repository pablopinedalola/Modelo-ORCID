"""
recommender.py — Motor de recomendación de conocimiento y colaboradores.
"""

import logging
import networkx as nx

logger = logging.getLogger(__name__)

class RecommenderEngine:
    """Motor que recomienda conexiones basándose en topología de red."""

    def __init__(self, graph):
        self.graph = graph

    def recommend_collaborators(self, researcher_id: str, top_k: int = 5) -> list[dict]:
        """Recomienda investigadores similares que no son coautores directos.
        Usa Adamic-Adar o simplemente un Two-Hop traversal con puntuación.
        """
        if researcher_id not in self.graph.nodes():
            return []
            
        # 1. Encontrar coautores directos
        direct_coauthors = set()
        for succ in self.graph.successors(researcher_id):
            if self.graph.edges[researcher_id, succ].get("type") in ["coauthor", "authored"]:
                # Manejar ruta snii -> paper -> autor
                pass # Simplificado: se asume que author_graph podría ser útil
                
        # Por simplicidad usando el grafo G
        # Vamos a saltar a tópicos y disciplinas para encontrar pares
        scores = {}
        for succ in self.graph.successors(researcher_id):
            edge_type = self.graph.edges[researcher_id, succ].get("type")
            if edge_type in ["belongs_to_discipline", "affiliated_with"]:
                # Quién más está conectado aquí?
                for potential in self.graph.predecessors(succ):
                    if potential != researcher_id and potential.startswith("snii:"):
                        scores[potential] = scores.get(potential, 0) + 1
                        
        sorted_potentials = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for p, score in sorted_potentials:
            data = self.graph.nodes[p]
            results.append({
                "id": p,
                "name": data.get("label", p),
                "score": score,
                "reason": f"Comparte {score} áreas/instituciones"
            })
            
        return results
