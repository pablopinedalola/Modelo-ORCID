"""
neighbor_retriever.py — Extracción de subgrafos vecinales.

Realiza traversal del Knowledge Graph desde nodos iniciales
para descubrir entidades relacionadas en 1, 2 o N saltos.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from src.graph.knowledge_graph import AcademicKnowledgeGraph

logger = logging.getLogger(__name__)

class NeighborRetriever:
    """Explorador de vecindarios académicos.
    
    Extrae subgrafos y listas de nodos conectados a un conjunto de nodos origen,
    utilizando pesos y filtrado por tipos de arista.
    """
    
    def __init__(self, graph: AcademicKnowledgeGraph):
        self.graph = graph

    def get_multi_hop_neighbors(
        self, 
        seed_nodes: list[str], 
        max_depth: int = 2,
        edge_types: Optional[list[str]] = None
    ) -> dict[str, float]:
        """Obtiene vecinos hasta cierta profundidad con atenuación de peso.
        
        Args:
            seed_nodes: Nodos semilla (ej. ['snii:123', 'snii:456']).
            max_depth: Profundidad máxima del traversal.
            edge_types: Aristas permitidas. Si None, todas.
            
        Returns:
            Dict de {node_id: proximity_score}
        """
        if not self.graph or not self.graph.G:
            return {}
            
        G = self.graph.G
        proximity = {}
        
        # Iniciar semillas con proximidad 1.0
        frontier = {node: 1.0 for node in seed_nodes if node in G}
        proximity.update(frontier)
        
        visited = set(frontier.keys())
        
        for depth in range(1, max_depth + 1):
            next_frontier = {}
            for node, current_score in frontier.items():
                for succ in G.successors(node):
                    if succ in visited:
                        continue
                        
                    edge_data = G.edges[node, succ]
                    edge_type = edge_data.get("type")
                    if edge_types and edge_type not in edge_types:
                        continue
                        
                    weight = edge_data.get("weight", 1.0)
                    # Atenuación por distancia (decay)
                    decay = 0.5 ** depth
                    score = current_score * weight * decay
                    
                    if succ not in next_frontier or score > next_frontier[succ]:
                        next_frontier[succ] = score
                        
                for pred in G.predecessors(node):
                    if pred in visited:
                        continue
                        
                    edge_data = G.edges[pred, node]
                    edge_type = edge_data.get("type")
                    if edge_types and edge_type not in edge_types:
                        continue
                        
                    weight = edge_data.get("weight", 1.0)
                    decay = 0.5 ** depth
                    score = current_score * weight * decay
                    
                    if pred not in next_frontier or score > next_frontier[pred]:
                        next_frontier[pred] = score
                        
            frontier = next_frontier
            for n, s in frontier.items():
                if n not in proximity or s > proximity[n]:
                    proximity[n] = s
            visited.update(frontier.keys())
            
        return proximity
