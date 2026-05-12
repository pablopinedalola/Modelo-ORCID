"""
ambiguity_analysis.py -- Analisis de ambiguedad e incertidumbre.

Detecta:
- Casos ambiguos (candidatos con scores muy cercanos).
- Incertidumbre residual (dimensiones sin evidencia).
- Entropia de la distribucion de confidence.
- Confidence gap entre el mejor y segundo mejor candidato.

Usage:
    analyzer = AmbiguityAnalyzer()
    report = analyzer.analyze(trace)
    print(f"Ambiguity: {report.ambiguity_score:.3f}")
    print(f"Entropy: {report.entropy:.3f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.interpreter.evidence_trace import (
    EvidenceTrace,
    DIMENSION_NAMES,
)


@dataclass
class AmbiguityReport:
    """Reporte de ambiguedad para un investigador.

    Attributes:
        researcher_id: ID del investigador.
        researcher_name: Nombre.
        ambiguity_score: Score de ambiguedad [0=claro, 1=ambiguo].
        confidence_gap: Diferencia entre 1er y 2do candidato.
        entropy: Entropia de la distribucion de confidence.
        undecided_dimensions: Dimensiones sin evidencia significativa.
        competing_candidates: Candidatos con scores muy cercanos.
        dominant_candidate_ratio: Ratio de dominancia del mejor candidato.
        risk_level: "low" | "medium" | "high"
    """
    researcher_id: str = ""
    researcher_name: str = ""
    ambiguity_score: float = 0.0
    confidence_gap: float = 0.0
    entropy: float = 0.0
    undecided_dimensions: list[str] = field(default_factory=list)
    competing_candidates: list[dict] = field(default_factory=list)
    dominant_candidate_ratio: float = 0.0
    risk_level: str = "low"
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "researcher_id": self.researcher_id,
            "researcher_name": self.researcher_name,
            "ambiguity_score": round(self.ambiguity_score, 4),
            "confidence_gap": round(self.confidence_gap, 4),
            "entropy": round(self.entropy, 4),
            "undecided_dimensions": self.undecided_dimensions,
            "competing_candidates": self.competing_candidates,
            "dominant_candidate_ratio": round(self.dominant_candidate_ratio, 4),
            "risk_level": self.risk_level,
            "details": self.details,
        }


class AmbiguityAnalyzer:
    """Analiza ambiguedad e incertidumbre en el refinamiento.

    Implementa multiples metricas de ambiguedad:

    1. Confidence Gap: gap = conf_best - conf_second
       Bajo gap = alta ambiguedad.

    2. Ambiguity Score: 1 - gap (normalizado)
       Alto = mas ambiguo.

    3. Entropia de Shannon sobre distribucion de confidence:
       H = -sum(p_i * log(p_i))
       Alta entropia = mas ambiguedad.

    4. Dimensiones indecisas: dimensiones con scores < umbral
       para el mejor candidato.

    Usage:
        analyzer = AmbiguityAnalyzer()
        report = analyzer.analyze(trace)
    """

    UNDECIDED_THRESHOLD = 0.3
    COMPETING_THRESHOLD = 0.1  # Gap < 0.1 = competing

    def analyze(self, trace: EvidenceTrace) -> AmbiguityReport:
        """Analiza la ambiguedad de un caso de refinamiento.

        Args:
            trace: Traza de evidencia completa.

        Returns:
            AmbiguityReport con metricas de ambiguedad.
        """
        report = AmbiguityReport(
            researcher_id=trace.researcher_id,
            researcher_name=trace.researcher_name,
        )

        # Get final confidences for all candidates
        traces = trace.get_all_candidate_traces()
        if not traces:
            report.ambiguity_score = 1.0
            report.risk_level = "high"
            report.details = "No hay candidatos para analizar."
            return report

        final_confs = []
        for cid, ct in traces.items():
            if ct.trajectory:
                final = ct.trajectory[-1]
                final_confs.append({
                    "candidate_id": cid,
                    "display_name": ct.display_name,
                    "confidence": final.confidence,
                    "scores": final.scores,
                })

        if not final_confs:
            report.ambiguity_score = 1.0
            report.risk_level = "high"
            return report

        # Sort by confidence descending
        final_confs.sort(key=lambda x: x["confidence"], reverse=True)

        best = final_confs[0]
        second = final_confs[1] if len(final_confs) > 1 else None

        # 1. Confidence gap
        if second:
            report.confidence_gap = best["confidence"] - second["confidence"]
        else:
            report.confidence_gap = best["confidence"]

        # 2. Ambiguity score
        report.ambiguity_score = max(
            0.0,
            1.0 - report.confidence_gap
        )

        # 3. Entropy
        report.entropy = self._compute_entropy(
            [c["confidence"] for c in final_confs]
        )

        # 4. Undecided dimensions
        best_scores = best.get("scores", {})
        for dim in DIMENSION_NAMES:
            val = best_scores.get(dim, 0.0)
            if val < self.UNDECIDED_THRESHOLD:
                report.undecided_dimensions.append(dim)

        # 5. Competing candidates
        if second and report.confidence_gap < self.COMPETING_THRESHOLD:
            report.competing_candidates = [
                {
                    "id": c["candidate_id"],
                    "name": c["display_name"],
                    "confidence": round(c["confidence"], 4),
                }
                for c in final_confs[:3]
            ]

        # 6. Dominant candidate ratio
        total_conf = sum(c["confidence"] for c in final_confs)
        if total_conf > 0:
            report.dominant_candidate_ratio = (
                best["confidence"] / total_conf
            )

        # 7. Risk level
        report.risk_level = self._classify_risk(report)

        # 8. Build details text
        report.details = self._build_details(report, best, second)

        return report

    def batch_analyze(
        self,
        traces: list[EvidenceTrace],
    ) -> list[AmbiguityReport]:
        """Analiza multiples trazas y retorna reportes.

        Args:
            traces: Lista de trazas.

        Returns:
            Lista de AmbiguityReports.
        """
        return [self.analyze(t) for t in traces]

    def summary_stats(
        self,
        reports: list[AmbiguityReport],
    ) -> dict:
        """Calcula estadisticas agregadas de ambiguedad.

        Args:
            reports: Lista de reportes.

        Returns:
            Dict con metricas agregadas.
        """
        if not reports:
            return {}

        n = len(reports)
        return {
            "total_cases": n,
            "avg_ambiguity": sum(r.ambiguity_score for r in reports) / n,
            "avg_entropy": sum(r.entropy for r in reports) / n,
            "avg_confidence_gap": sum(r.confidence_gap for r in reports) / n,
            "high_risk_count": sum(
                1 for r in reports if r.risk_level == "high"
            ),
            "medium_risk_count": sum(
                1 for r in reports if r.risk_level == "medium"
            ),
            "low_risk_count": sum(
                1 for r in reports if r.risk_level == "low"
            ),
            "most_undecided_dims": self._most_common_undecided(reports),
        }

    # ── Private Helpers ───────────────────────────────────────────────

    def _compute_entropy(self, confidences: list[float]) -> float:
        """Calcula entropia de Shannon de la distribucion de confidence.

        H = -sum(p_i * log2(p_i))

        donde p_i = conf_i / sum(conf).
        """
        total = sum(confidences)
        if total <= 0:
            return 0.0

        probs = [c / total for c in confidences if c > 0]
        if len(probs) <= 1:
            return 0.0

        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log2(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _classify_risk(self, report: AmbiguityReport) -> str:
        """Clasifica el nivel de riesgo de ambiguedad."""
        if report.ambiguity_score >= 0.8 or len(report.undecided_dimensions) >= 4:
            return "high"
        if report.ambiguity_score >= 0.5 or len(report.undecided_dimensions) >= 2:
            return "medium"
        return "low"

    def _build_details(
        self,
        report: AmbiguityReport,
        best: dict,
        second: Optional[dict],
    ) -> str:
        """Construye texto descriptivo de la ambiguedad."""
        lines = []

        if report.risk_level == "high":
            lines.append(
                f"ALTO RIESGO de ambiguedad para {report.researcher_name}."
            )
        elif report.risk_level == "medium":
            lines.append(
                f"Ambiguedad MODERADA para {report.researcher_name}."
            )
        else:
            lines.append(
                f"Ambiguedad BAJA para {report.researcher_name}."
            )

        lines.append(
            f"Mejor candidato: {best['display_name']} "
            f"(conf={best['confidence']:.3f})."
        )

        if second:
            lines.append(
                f"Segundo candidato: {second['display_name']} "
                f"(conf={second['confidence']:.3f})."
            )
            lines.append(f"Gap: {report.confidence_gap:.3f}")

        if report.undecided_dimensions:
            dims = ", ".join(report.undecided_dimensions)
            lines.append(f"Dimensiones sin evidencia: {dims}.")

        return " ".join(lines)

    def _most_common_undecided(
        self,
        reports: list[AmbiguityReport],
    ) -> dict[str, int]:
        """Encuentra dimensiones mas frecuentemente indecisas."""
        counts: dict[str, int] = {}
        for r in reports:
            for d in r.undecided_dimensions:
                counts[d] = counts.get(d, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
