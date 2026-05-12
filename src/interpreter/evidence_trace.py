"""
evidence_trace.py -- Sistema de trazabilidad de evidencia.

Registra la evolucion completa del refinement engine:
cada iteracion, cada candidato, cada dimension, cada vecino.

Permite reconstruir exactamente por que el sistema tomo
cada decision, conectando el comportamiento computacional
con el modelo matematico:

    d_{n+1}(v) = F(d_n(v))
    F(d)(v) = d(v) V V_{u in N(v)} phi(d(u))

Usage:
    trace = EvidenceTrace(researcher_id="snii:abc123")
    trace.begin_iteration(1)
    trace.record_candidate_state(cand_id, evidence_vector)
    trace.record_evidence_from_neighbor(cand_id, neighbor_id, evidence)
    trace.end_iteration(converged=False, max_delta=0.15)
    ...
    trace.finalize(best_candidate_id="cand:openalex:A123")
    report = trace.to_dict()
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── Data Structures ──────────────────────────────────────────────────────────

DIMENSION_NAMES = [
    "name", "institution", "area",
    "publication", "coauthor", "temporal", "semantic",
]


@dataclass
class EvidenceContribution:
    """Registro de una contribucion de evidencia de un vecino.

    Corresponde a phi(u) en el modelo matematico:
    la funcion de extraccion de evidencia aplicada al vecino u.

    Attributes:
        neighbor_id: ID del nodo vecino (u).
        neighbor_type: Tipo de nodo (paper, institution, etc.).
        edge_type: Tipo de arista que los conecta.
        dimension: Dimension de evidencia afectada.
        value: Valor de la evidencia extraida.
        previous_value: Valor anterior de esa dimension.
        was_applied: Si la evidencia cambio algo (combine via max).
        detail: Descripcion legible de la evidencia.
    """
    neighbor_id: str
    neighbor_type: str
    edge_type: str
    dimension: str
    value: float
    previous_value: float
    was_applied: bool = False
    detail: str = ""

    @property
    def delta(self) -> float:
        """Cambio neto de esta contribucion."""
        return self.value - self.previous_value if self.was_applied else 0.0


@dataclass
class CandidateSnapshot:
    """Estado de un candidato en un instante dado.

    Corresponde a d_n(v) en el modelo matematico:
    el descriptor del candidato v en la iteracion n.

    Attributes:
        candidate_id: ID del candidato.
        display_name: Nombre para visualizacion.
        scores: Dict {dimension: score} en este instante.
        confidence: Score de confianza agregado.
    """
    candidate_id: str
    display_name: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IterationSnapshot:
    """Registro completo de una iteracion del refinamiento.

    Corresponde a la aplicacion de F al estado actual:
    d_{n+1} = F(d_n).

    Attributes:
        iteration: Numero de iteracion (1-indexed).
        candidate_states_before: Estado de cada candidato ANTES.
        candidate_states_after: Estado de cada candidato DESPUES.
        contributions: Todas las contribuciones de evidencia.
        max_delta: Maximo cambio absoluto en confidence.
        converged: Si se alcanzo convergencia.
        dominant_dimension: Dimension que mas cambio.
        total_new_evidence: Numero de contribuciones que aportaron algo.
    """
    iteration: int
    candidate_states_before: dict[str, CandidateSnapshot] = field(
        default_factory=dict
    )
    candidate_states_after: dict[str, CandidateSnapshot] = field(
        default_factory=dict
    )
    contributions: list[EvidenceContribution] = field(default_factory=list)
    max_delta: float = 0.0
    converged: bool = False
    dominant_dimension: str = ""
    total_new_evidence: int = 0

    def compute_summary(self) -> None:
        """Calcula metricas de resumen de la iteracion."""
        applied = [c for c in self.contributions if c.was_applied]
        self.total_new_evidence = len(applied)

        # Calcular dimension dominante (mas cambio acumulado)
        dim_deltas: dict[str, float] = {}
        for c in applied:
            dim_deltas[c.dimension] = (
                dim_deltas.get(c.dimension, 0.0) + abs(c.delta)
            )
        if dim_deltas:
            self.dominant_dimension = max(dim_deltas, key=dim_deltas.get)

    def to_dict(self) -> dict:
        d = {
            "iteration": self.iteration,
            "max_delta": round(self.max_delta, 6),
            "converged": self.converged,
            "dominant_dimension": self.dominant_dimension,
            "total_new_evidence": self.total_new_evidence,
            "candidate_states_before": {
                k: v.to_dict()
                for k, v in self.candidate_states_before.items()
            },
            "candidate_states_after": {
                k: v.to_dict()
                for k, v in self.candidate_states_after.items()
            },
            "contributions": [asdict(c) for c in self.contributions],
        }
        return d


@dataclass
class CandidateTrace:
    """Traza completa de un candidato a traves de todas las iteraciones.

    Permite ver la trayectoria d_0(v), d_1(v), ..., d*(v)
    de un candidato especifico.

    Attributes:
        candidate_id: ID del candidato.
        display_name: Nombre para visualizacion.
        trajectory: Lista de snapshots ordenados por iteracion.
    """
    candidate_id: str
    display_name: str = ""
    trajectory: list[CandidateSnapshot] = field(default_factory=list)

    @property
    def initial_confidence(self) -> float:
        return self.trajectory[0].confidence if self.trajectory else 0.0

    @property
    def final_confidence(self) -> float:
        return self.trajectory[-1].confidence if self.trajectory else 0.0

    @property
    def confidence_gain(self) -> float:
        return self.final_confidence - self.initial_confidence

    def dimension_trajectory(self, dimension: str) -> list[float]:
        """Retorna la trayectoria de una dimension especifica."""
        return [s.scores.get(dimension, 0.0) for s in self.trajectory]

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "initial_confidence": round(self.initial_confidence, 4),
            "final_confidence": round(self.final_confidence, 4),
            "confidence_gain": round(self.confidence_gain, 4),
            "trajectory": [s.to_dict() for s in self.trajectory],
        }


# ─── Main Trace Class ────────────────────────────────────────────────────────

class EvidenceTrace:
    """Registro completo de la evolucion del refinement engine.

    Captura toda la informacion necesaria para:
    - Explicar decisiones
    - Visualizar evolucion
    - Analizar dinamica
    - Conectar con modelo matematico

    Usage:
        trace = EvidenceTrace("snii:abc123", "Carlos Garcia")
        trace.record_initial_state(candidates)
        trace.begin_iteration(1)
        ...
        trace.end_iteration(converged=False, max_delta=0.15)
        trace.finalize("cand:openalex:A123")

    Attributes:
        researcher_id: ID del investigador SNII.
        researcher_name: Nombre del investigador.
        iterations: Lista de snapshots de cada iteracion.
        timestamp: Momento de creacion.
    """

    def __init__(
        self,
        researcher_id: str,
        researcher_name: str = "",
    ) -> None:
        self.researcher_id = researcher_id
        self.researcher_name = researcher_name
        self.iterations: list[IterationSnapshot] = []
        self.initial_states: dict[str, CandidateSnapshot] = {}
        self.best_candidate_id: Optional[str] = None
        self.total_iterations: int = 0
        self.converged: bool = False
        self.timestamp: str = datetime.now().isoformat()

        # Working state
        self._current_iteration: Optional[IterationSnapshot] = None

    def record_initial_state(
        self,
        candidate_id: str,
        display_name: str,
        scores: dict[str, float],
        confidence: float,
    ) -> None:
        """Registra estado inicial d_0(v) de un candidato."""
        snap = CandidateSnapshot(
            candidate_id=candidate_id,
            display_name=display_name,
            scores=dict(scores),
            confidence=round(confidence, 6),
        )
        self.initial_states[candidate_id] = snap

    def begin_iteration(self, iteration: int) -> None:
        """Inicia una nueva iteracion del refinamiento."""
        self._current_iteration = IterationSnapshot(iteration=iteration)

    def record_candidate_state_before(
        self,
        candidate_id: str,
        display_name: str,
        scores: dict[str, float],
        confidence: float,
    ) -> None:
        """Registra estado de un candidato ANTES de la iteracion."""
        if self._current_iteration is None:
            return
        snap = CandidateSnapshot(
            candidate_id=candidate_id,
            display_name=display_name,
            scores=dict(scores),
            confidence=round(confidence, 6),
        )
        self._current_iteration.candidate_states_before[candidate_id] = snap

    def record_candidate_state_after(
        self,
        candidate_id: str,
        display_name: str,
        scores: dict[str, float],
        confidence: float,
    ) -> None:
        """Registra estado de un candidato DESPUES de la iteracion."""
        if self._current_iteration is None:
            return
        snap = CandidateSnapshot(
            candidate_id=candidate_id,
            display_name=display_name,
            scores=dict(scores),
            confidence=round(confidence, 6),
        )
        self._current_iteration.candidate_states_after[candidate_id] = snap

    def record_evidence_contribution(
        self,
        candidate_id: str,
        neighbor_id: str,
        neighbor_type: str,
        edge_type: str,
        dimension: str,
        value: float,
        previous_value: float,
        detail: str = "",
    ) -> None:
        """Registra una contribucion de evidencia phi(u) para un candidato.

        Args:
            candidate_id: Candidato que recibe la evidencia.
            neighbor_id: Vecino que la aporta (u in N(v)).
            neighbor_type: Tipo de nodo vecino.
            edge_type: Tipo de arista.
            dimension: Dimension de evidencia afectada.
            value: Valor extraido por phi(u).
            previous_value: Valor anterior de esa dimension.
            detail: Descripcion legible.
        """
        if self._current_iteration is None:
            return

        was_applied = value > previous_value
        contribution = EvidenceContribution(
            neighbor_id=neighbor_id,
            neighbor_type=neighbor_type,
            edge_type=edge_type,
            dimension=dimension,
            value=round(value, 6),
            previous_value=round(previous_value, 6),
            was_applied=was_applied,
            detail=detail,
        )
        self._current_iteration.contributions.append(contribution)

    def end_iteration(
        self,
        converged: bool,
        max_delta: float,
    ) -> None:
        """Finaliza la iteracion actual."""
        if self._current_iteration is None:
            return
        self._current_iteration.converged = converged
        self._current_iteration.max_delta = round(max_delta, 6)
        self._current_iteration.compute_summary()
        self.iterations.append(self._current_iteration)
        self._current_iteration = None

    def finalize(
        self,
        best_candidate_id: Optional[str],
        converged: bool,
    ) -> None:
        """Finaliza la traza completa del refinamiento."""
        self.best_candidate_id = best_candidate_id
        self.converged = converged
        self.total_iterations = len(self.iterations)

    # ── Extraction Methods ────────────────────────────────────────────

    def get_candidate_trace(self, candidate_id: str) -> CandidateTrace:
        """Extrae la traza completa de un candidato individual.

        Retorna la trayectoria d_0(v), d_1(v), ..., d*(v).
        """
        ct = CandidateTrace(candidate_id=candidate_id)

        # Estado inicial
        if candidate_id in self.initial_states:
            init = self.initial_states[candidate_id]
            ct.display_name = init.display_name
            ct.trajectory.append(copy.deepcopy(init))

        # Estados por iteracion (after)
        for it in self.iterations:
            if candidate_id in it.candidate_states_after:
                ct.trajectory.append(
                    copy.deepcopy(it.candidate_states_after[candidate_id])
                )

        return ct

    def get_all_candidate_traces(self) -> dict[str, CandidateTrace]:
        """Extrae trazas de todos los candidatos."""
        traces = {}
        for cid in self.initial_states:
            traces[cid] = self.get_candidate_trace(cid)
        return traces

    def get_contributions_for(
        self,
        candidate_id: str,
        iteration: Optional[int] = None,
    ) -> list[EvidenceContribution]:
        """Obtiene contribuciones de evidencia para un candidato."""
        result = []
        for it in self.iterations:
            if iteration is not None and it.iteration != iteration:
                continue
            for c in it.contributions:
                # Contributions are linked by the neighbor providing to this cand
                # We check if the candidate's state changed
                result.append(c)
        return result

    def confidence_trajectory(self, candidate_id: str) -> list[float]:
        """Retorna la trayectoria de confidence de un candidato."""
        trace = self.get_candidate_trace(candidate_id)
        return [s.confidence for s in trace.trajectory]

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializa la traza completa a dict."""
        return {
            "researcher_id": self.researcher_id,
            "researcher_name": self.researcher_name,
            "timestamp": self.timestamp,
            "total_iterations": self.total_iterations,
            "converged": self.converged,
            "best_candidate_id": self.best_candidate_id,
            "initial_states": {
                k: v.to_dict() for k, v in self.initial_states.items()
            },
            "iterations": [it.to_dict() for it in self.iterations],
        }

    def save_json(self, filepath: str | Path) -> Path:
        """Guarda la traza como JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceTrace":
        """Reconstruye una traza desde dict."""
        trace = cls(
            researcher_id=data["researcher_id"],
            researcher_name=data.get("researcher_name", ""),
        )
        trace.timestamp = data.get("timestamp", "")
        trace.total_iterations = data.get("total_iterations", 0)
        trace.converged = data.get("converged", False)
        trace.best_candidate_id = data.get("best_candidate_id")

        for cid, snap_data in data.get("initial_states", {}).items():
            trace.initial_states[cid] = CandidateSnapshot(
                candidate_id=snap_data["candidate_id"],
                display_name=snap_data.get("display_name", ""),
                scores=snap_data.get("scores", {}),
                confidence=snap_data.get("confidence", 0),
            )
        return trace
