"""
trend_detection.py — Detección de tendencias y disciplinas emergentes.
"""

import logging
from collections import defaultdict
import numpy as np

from src.analytics.temporal_analysis import TemporalAnalyzer

logger = logging.getLogger(__name__)

class TrendDetector:
    """Detecta áreas de rápido crecimiento o decaimiento."""

    def __init__(self, graph):
        self.temporal = TemporalAnalyzer(graph)

    def detect_trending_topics(self) -> dict[str, dict]:
        """Calcula la pendiente de crecimiento de los tópicos en los últimos 5 años."""
        evolution = self.temporal.get_topic_evolution()
        
        trends = {}
        # Asumimos año máximo 2026 para el análisis
        recent_years = [2021, 2022, 2023, 2024, 2025, 2026]
        
        for topic, years_data in evolution.items():
            counts = [years_data.get(y, 0) for y in recent_years]
            if sum(counts) < 3: # Ignorar tópicos con muy bajo volumen reciente
                continue
                
            # Regresión lineal simple para detectar pendiente
            x = np.arange(len(recent_years))
            y = np.array(counts)
            slope, intercept = np.polyfit(x, y, 1)
            
            status = "stable"
            if slope > 0.5:
                status = "emerging"
            elif slope < -0.5:
                status = "declining"
                
            trends[topic] = {
                "slope": round(float(slope), 3),
                "total_recent": sum(counts),
                "status": status,
                "counts": counts
            }
            
        # Ordenar por pendiente de crecimiento (slope)
        return dict(sorted(trends.items(), key=lambda item: item[1]["slope"], reverse=True))
