"""
evidence_propagation.py — Propagación de evidencia para retrieval.

Implementa el modelo matemático original adaptado a búsqueda:
F(d)(v) = d(v) ∨ ⋁_{u ∈ N(v)} φ(d(u))

Donde la evidencia d(v) es el retrieval score inicial, y φ es la 
transmisión de score moderada por el peso de la conexión.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import networkx as nx

from src.graph.knowledge_graph import AcademicKnowledgeGraph

logger = logging.getLogger(__name__)

class EvidencePropagation:
    """Motor de propagación de evidencia sobre el Knowledge Graph."""

    def __init__(self, graph: AcademicKnowledgeGraph):
        self.graph = graph

    def propagate(
        self, 
        initial_scores: dict[str, float], 
        iterations: int = 2,
        decay_factor: float = 0.5
    ) -> dict[str, float]:
        """Propaga scores a través del grafo.
        
        Args:
            initial_scores: Dict {node_id: initial_score}
            iterations: Número de iteraciones (n).
            decay_factor: Factor de atenuación al propagar por aristas.
            
        Returns:
            Dict con los scores refinados tras propagación.
        """
        if not self.graph or not self.graph.G:
            return initial_scores

        G = self.graph.G
        scores = deepcopy(initial_scores)
        
        # Filtrar solo nodos que existen en el grafo
        scores = {k: v for k, v in scores.items() if k in G}
        
        # Historial de explicaciones de propagación
        provenance = {k: [] for k in scores}

        for n in range(iterations):
            new_scores = deepcopy(scores)
            
            # Para cada nodo que tiene score, propagar a vecinos
            for u, score_u in scores.items():
                if score_u <= 0.05:  # Optimización: ignorar ruido bajo
                    continue
                    
                # Vecinos salientes
                for v in G.successors(u):
                    weight = G.edges[u, v].get("weight", 1.0)
                    phi_u = score_u * weight * decay_factor
                    
                    if v not in new_scores:
                        new_scores[v] = 0.0
                        provenance[v] = []
                        
                    # Operador Lógica V (Lattice Join -> Max)
                    if phi_u > new_scores[v]:
                        new_scores[v] = phi_u
                        provenance[v].append((u, G.edges[u, v].get("type", "connected")))

                # Vecinos entrantes
                for v in G.predecessors(u):
                    weight = G.edges[v, u].get("weight", 1.0)
                    phi_u = score_u * weight * decay_factor
                    
                    if v not in new_scores:
                        new_scores[v] = 0.0
                        provenance[v] = []
                        
                    if phi_u > new_scores[v]:
                        new_scores[v] = phi_u
                        provenance[v].append((u, G.edges[v, u].get("type", "connected")))

            scores = new_scores

        return scores, provenance
