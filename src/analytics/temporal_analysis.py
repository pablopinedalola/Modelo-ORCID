"""
temporal_analysis.py — Análisis temporal de producción científica.

Modela crecimiento, decadencia y migraciones temáticas.
"""

import logging
from collections import defaultdict
import networkx as nx

from src.models.schemas import NodeType

logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """Analizador de evolución científica en el tiempo."""

    def __init__(self, graph):
        self.graph = graph

    def get_papers_by_year(self) -> dict[int, int]:
        """Calcula la distribución de producción anual."""
        distribution = defaultdict(int)
        for n, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.PAPER.value:
                year = data.get("year")
                if year and isinstance(year, int) and 1950 < year <= 2030:
                    distribution[year] += 1
        return dict(sorted(distribution.items()))

    def get_topic_evolution(self) -> dict[str, dict[int, int]]:
        """Mide cómo evolucionan los tópicos por año."""
        topic_years = defaultdict(lambda: defaultdict(int))
        
        for paper, data in self.graph.nodes(data=True):
            if data.get("type") == NodeType.PAPER.value:
                year = data.get("year")
                if not year or not isinstance(year, int) or year < 1950:
                    continue
                    
                # Encontrar topics de este paper
                for succ in self.graph.successors(paper):
                    succ_data = self.graph.nodes[succ]
                    if succ_data.get("type") == NodeType.TOPIC.value:
                        topic_name = succ_data.get("label", "Unknown")
                        topic_years[topic_name][year] += 1
                        
        # Formatear
        result = {}
        for topic, years in topic_years.items():
            result[topic] = dict(sorted(years.items()))
            
        return result
