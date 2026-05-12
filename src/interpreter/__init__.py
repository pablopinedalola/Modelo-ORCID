"""
Modulo de interpretabilidad y trazabilidad del refinement engine.

Proporciona:
    - EvidenceTrace: registro detallado de toda la evolucion del refinamiento.
    - StateVisualizer: visualizacion de la evolucion d_0 -> d_1 -> ... -> d*.
    - MatchExplainer: explicaciones en lenguaje natural de decisiones.
    - AmbiguityAnalyzer: deteccion de casos ambiguos e incertidumbre.
    - DynamicsAnalyzer: analisis de propagacion, estabilidad y atractores.
"""

from .evidence_trace import EvidenceTrace, IterationSnapshot, CandidateTrace
from .explainer import MatchExplainer
from .ambiguity_analysis import AmbiguityAnalyzer
from .dynamics import DynamicsAnalyzer
from .state_visualizer import StateVisualizer

__all__ = [
    "EvidenceTrace",
    "IterationSnapshot",
    "CandidateTrace",
    "MatchExplainer",
    "AmbiguityAnalyzer",
    "DynamicsAnalyzer",
    "StateVisualizer",
]
