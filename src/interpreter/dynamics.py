"""
dynamics.py -- Analisis de dinamica del refinamiento.

Estudia:
- Propagacion de evidencia a traves del grafo.
- Estabilidad del proceso iterativo.
- Atractores (puntos fijos a los que converge el sistema).
- Saturacion de dimensiones.
- Dominancia de dimensiones.
- Velocidad de convergencia.

Conecta el comportamiento computacional con el modelo matematico:
    d_{n+1}(v) = F(d_n(v))

y analiza las propiedades de F como operador de refinamiento.

Usage:
    analyzer = DynamicsAnalyzer()
    report = analyzer.analyze(trace)
    print(report.to_dict())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.interpreter.evidence_trace import (
    EvidenceTrace,
    CandidateTrace,
    DIMENSION_NAMES,
)


@dataclass
class DimensionDynamics:
    """Dinamica de una dimension individual.

    Attributes:
        dimension: Nombre de la dimension.
        initial_value: Valor inicial d_0.
        final_value: Valor final d*.
        total_gain: Ganancia total = final - initial.
        saturation_iteration: Iteracion donde se satura (deja de crecer).
        is_saturated: Si la dimension alcanzo su maximo.
        contribution_count: Numero de contribuciones de evidencia.
        growth_rate: Tasa de crecimiento promedio por iteracion.
    """
    dimension: str = ""
    initial_value: float = 0.0
    final_value: float = 0.0
    total_gain: float = 0.0
    saturation_iteration: int = 0
    is_saturated: bool = False
    contribution_count: int = 0
    growth_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "initial_value": round(self.initial_value, 4),
            "final_value": round(self.final_value, 4),
            "total_gain": round(self.total_gain, 4),
            "saturation_iteration": self.saturation_iteration,
            "is_saturated": self.is_saturated,
            "contribution_count": self.contribution_count,
            "growth_rate": round(self.growth_rate, 6),
        }


@dataclass
class ConvergenceProfile:
    """Perfil de convergencia del proceso iterativo.

    Attributes:
        converged: Si convergio.
        total_iterations: Iteraciones ejecutadas.
        convergence_rate: Tasa de convergencia (reduction ratio).
        deltas: Lista de max_delta por iteracion.
        is_monotonic: Si los deltas decrecen monotonicamente.
        oscillation_count: Numero de oscilaciones en delta.
        lyapunov_estimate: Estimacion del exponente de Lyapunov.
    """
    converged: bool = False
    total_iterations: int = 0
    convergence_rate: float = 0.0
    deltas: list[float] = field(default_factory=list)
    is_monotonic: bool = True
    oscillation_count: int = 0
    lyapunov_estimate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "total_iterations": self.total_iterations,
            "convergence_rate": round(self.convergence_rate, 6),
            "deltas": [round(d, 6) for d in self.deltas],
            "is_monotonic": self.is_monotonic,
            "oscillation_count": self.oscillation_count,
            "lyapunov_estimate": round(self.lyapunov_estimate, 6),
        }


@dataclass
class PropagationAnalysis:
    """Analisis de propagacion de evidencia en el grafo.

    Attributes:
        total_contributions: Total de contribuciones de evidencia.
        applied_contributions: Contribuciones que aportaron nueva info.
        application_rate: Fraccion de contribuciones efectivas.
        evidence_by_type: Contribuciones agrupadas por tipo de vecino.
        most_influential_neighbors: Vecinos mas influyentes.
    """
    total_contributions: int = 0
    applied_contributions: int = 0
    application_rate: float = 0.0
    evidence_by_type: dict[str, int] = field(default_factory=dict)
    most_influential_neighbors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_contributions": self.total_contributions,
            "applied_contributions": self.applied_contributions,
            "application_rate": round(self.application_rate, 4),
            "evidence_by_type": self.evidence_by_type,
            "most_influential_neighbors": self.most_influential_neighbors,
        }


@dataclass
class DynamicsReport:
    """Reporte completo de dinamica del refinamiento.

    Attributes:
        researcher_id: ID del investigador.
        researcher_name: Nombre.
        convergence: Perfil de convergencia.
        dimension_dynamics: Dinamica por dimension para el mejor candidato.
        propagation: Analisis de propagacion.
        attractor_type: Tipo de atractor detectado.
        dominant_dimension: Dimension mas influyente.
        stability_score: Score de estabilidad [0-1].
    """
    researcher_id: str = ""
    researcher_name: str = ""
    convergence: ConvergenceProfile = field(
        default_factory=ConvergenceProfile
    )
    dimension_dynamics: list[DimensionDynamics] = field(
        default_factory=list
    )
    propagation: PropagationAnalysis = field(
        default_factory=PropagationAnalysis
    )
    attractor_type: str = "unknown"
    dominant_dimension: str = ""
    stability_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "researcher_id": self.researcher_id,
            "researcher_name": self.researcher_name,
            "convergence": self.convergence.to_dict(),
            "dimension_dynamics": [d.to_dict() for d in self.dimension_dynamics],
            "propagation": self.propagation.to_dict(),
            "attractor_type": self.attractor_type,
            "dominant_dimension": self.dominant_dimension,
            "stability_score": round(self.stability_score, 4),
        }


class DynamicsAnalyzer:
    """Analiza la dinamica del proceso de refinamiento iterativo.

    Estudia el comportamiento del operador F como sistema dinamico:
    - Velocidad y tasa de convergencia
    - Saturacion y dominancia de dimensiones
    - Propagacion de evidencia
    - Tipo de atractor (fixed point, limit cycle, etc.)
    - Estabilidad del punto fijo

    Usage:
        analyzer = DynamicsAnalyzer()
        report = analyzer.analyze(trace)
    """

    SATURATION_EPSILON = 0.001

    def analyze(self, trace: EvidenceTrace) -> DynamicsReport:
        """Analiza la dinamica completa de un caso de refinamiento.

        Args:
            trace: Traza de evidencia.

        Returns:
            DynamicsReport con analisis completo.
        """
        report = DynamicsReport(
            researcher_id=trace.researcher_id,
            researcher_name=trace.researcher_name,
        )

        # 1. Convergence profile
        report.convergence = self._analyze_convergence(trace)

        # 2. Dimension dynamics (for best candidate)
        if trace.best_candidate_id:
            ct = trace.get_candidate_trace(trace.best_candidate_id)
            report.dimension_dynamics = self._analyze_dimensions(ct, trace)

        # 3. Propagation analysis
        report.propagation = self._analyze_propagation(trace)

        # 4. Attractor type
        report.attractor_type = self._classify_attractor(report.convergence)

        # 5. Dominant dimension
        if report.dimension_dynamics:
            best_dim = max(
                report.dimension_dynamics,
                key=lambda d: d.total_gain,
            )
            report.dominant_dimension = best_dim.dimension

        # 6. Stability score
        report.stability_score = self._compute_stability(report)

        return report

    def batch_analyze(
        self,
        traces: list[EvidenceTrace],
    ) -> list[DynamicsReport]:
        """Analiza multiples trazas."""
        return [self.analyze(t) for t in traces]

    # ── Convergence ───────────────────────────────────────────────────

    def _analyze_convergence(
        self,
        trace: EvidenceTrace,
    ) -> ConvergenceProfile:
        """Analiza el perfil de convergencia."""
        profile = ConvergenceProfile(
            converged=trace.converged,
            total_iterations=trace.total_iterations,
        )

        deltas = [it.max_delta for it in trace.iterations]
        profile.deltas = deltas

        if len(deltas) >= 2:
            # Convergence rate: ratio of reduction
            if deltas[0] > 0:
                profile.convergence_rate = (
                    deltas[-1] / deltas[0]
                )

            # Monotonicity check
            for i in range(1, len(deltas)):
                if deltas[i] > deltas[i - 1]:
                    profile.is_monotonic = False
                    profile.oscillation_count += 1

            # Lyapunov estimate (avg log ratio of consecutive deltas)
            log_ratios = []
            for i in range(1, len(deltas)):
                if deltas[i - 1] > 0 and deltas[i] > 0:
                    log_ratios.append(
                        math.log(deltas[i] / deltas[i - 1])
                    )
            if log_ratios:
                profile.lyapunov_estimate = (
                    sum(log_ratios) / len(log_ratios)
                )

        return profile

    # ── Dimension Dynamics ────────────────────────────────────────────

    def _analyze_dimensions(
        self,
        ct: CandidateTrace,
        trace: EvidenceTrace,
    ) -> list[DimensionDynamics]:
        """Analiza la dinamica de cada dimension para un candidato."""
        dynamics = []

        if not ct.trajectory:
            return dynamics

        for dim in DIMENSION_NAMES:
            traj = ct.dimension_trajectory(dim)
            if not traj:
                continue

            dd = DimensionDynamics(
                dimension=dim,
                initial_value=traj[0],
                final_value=traj[-1],
                total_gain=traj[-1] - traj[0],
            )

            # Saturation detection
            for i in range(1, len(traj)):
                if abs(traj[i] - traj[i - 1]) < self.SATURATION_EPSILON:
                    dd.saturation_iteration = i
                    dd.is_saturated = True
                    break

            if not dd.is_saturated and len(traj) > 1:
                dd.saturation_iteration = len(traj)

            # Growth rate
            if len(traj) >= 2:
                dd.growth_rate = dd.total_gain / (len(traj) - 1)

            # Count contributions for this dimension
            count = 0
            for it in trace.iterations:
                for c in it.contributions:
                    if c.dimension == dim and c.was_applied:
                        count += 1
            dd.contribution_count = count

            dynamics.append(dd)

        return dynamics

    # ── Propagation ───────────────────────────────────────────────────

    def _analyze_propagation(
        self,
        trace: EvidenceTrace,
    ) -> PropagationAnalysis:
        """Analiza la propagacion de evidencia."""
        analysis = PropagationAnalysis()

        for it in trace.iterations:
            analysis.total_contributions += len(it.contributions)
            for c in it.contributions:
                if c.was_applied:
                    analysis.applied_contributions += 1
                    # Count by neighbor type
                    analysis.evidence_by_type[c.neighbor_type] = (
                        analysis.evidence_by_type.get(c.neighbor_type, 0) + 1
                    )

        if analysis.total_contributions > 0:
            analysis.application_rate = (
                analysis.applied_contributions / analysis.total_contributions
            )

        # Most influential neighbors (by total delta contributed)
        neighbor_influence: dict[str, float] = {}
        for it in trace.iterations:
            for c in it.contributions:
                if c.was_applied:
                    neighbor_influence[c.neighbor_id] = (
                        neighbor_influence.get(c.neighbor_id, 0.0) + c.delta
                    )

        sorted_neighbors = sorted(
            neighbor_influence.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        analysis.most_influential_neighbors = [
            {"id": nid, "total_influence": round(inf, 4)}
            for nid, inf in sorted_neighbors[:5]
        ]

        return analysis

    # ── Attractor Classification ──────────────────────────────────────

    def _classify_attractor(
        self,
        convergence: ConvergenceProfile,
    ) -> str:
        """Clasifica el tipo de atractor del sistema.

        Tipos:
        - "fixed_point": Convergencia monotona a punto fijo.
        - "damped_oscillation": Convergencia con oscilaciones.
        - "limit_cycle": No converge, oscila.
        - "divergent": Deltas crecientes (inestable).
        - "immediate": Converge en 1-2 iteraciones.
        """
        if convergence.total_iterations <= 2 and convergence.converged:
            return "immediate"

        if convergence.converged:
            if convergence.is_monotonic:
                return "fixed_point"
            return "damped_oscillation"

        if convergence.oscillation_count > 2:
            return "limit_cycle"

        if convergence.lyapunov_estimate > 0:
            return "divergent"

        return "slow_convergence"

    # ── Stability ─────────────────────────────────────────────────────

    def _compute_stability(self, report: DynamicsReport) -> float:
        """Calcula score de estabilidad [0-1].

        Considera:
        - Si convergio (0.4)
        - Monotonia (0.2)
        - Baja tasa de convergencia (0.2)
        - Pocas oscilaciones (0.2)
        """
        score = 0.0

        if report.convergence.converged:
            score += 0.4

        if report.convergence.is_monotonic:
            score += 0.2

        # Low convergence rate is good (means fast convergence)
        cr = report.convergence.convergence_rate
        if cr < 0.1:
            score += 0.2
        elif cr < 0.5:
            score += 0.1

        # Few oscillations
        osc = report.convergence.oscillation_count
        if osc == 0:
            score += 0.2
        elif osc == 1:
            score += 0.1

        return min(score, 1.0)
