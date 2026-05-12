"""
explainer.py -- Motor de explicabilidad para decisiones del refinement engine.

Genera explicaciones en lenguaje natural de por que el sistema:
- asigno un ORCID a un investigador,
- rechazo un candidato,
- convergio en N iteraciones.

Tambien extrae reglas interpretables tipo decision tree para
hacer el sistema explicable ante investigadores y revisores.

Usage:
    explainer = MatchExplainer()
    text = explainer.explain_match(trace, candidate_id)
    rules = explainer.extract_rules(trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.interpreter.evidence_trace import (
    EvidenceTrace,
    CandidateTrace,
    DIMENSION_NAMES,
)


# ─── Dimension labels for explanations ────────────────────────────────────────

_DIM_LABELS = {
    "name": "similitud de nombre",
    "institution": "coincidencia institucional",
    "area": "coincidencia de area/disciplina",
    "publication": "evidencia de publicaciones",
    "coauthor": "red de coautoria",
    "temporal": "consistencia temporal",
    "semantic": "similitud semantica",
}

_DIM_LABELS_EN = {
    "name": "name similarity",
    "institution": "institutional match",
    "area": "area/discipline overlap",
    "publication": "publication evidence",
    "coauthor": "coauthor network",
    "temporal": "temporal consistency",
    "semantic": "semantic similarity",
}


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class InterpretableRule:
    """Regla interpretable extraida del comportamiento del sistema.

    Formato:
        IF condition_1 AND condition_2 ... THEN outcome

    Attributes:
        conditions: Lista de condiciones (string legible).
        outcome: Resultado predicho.
        confidence_range: (min, max) de confidence para esta regla.
        support: Cuantos casos soportan esta regla.
    """
    conditions: list[str] = field(default_factory=list)
    outcome: str = ""
    confidence_range: tuple[float, float] = (0.0, 1.0)
    support: int = 0

    def to_text(self) -> str:
        conds = "\n    AND ".join(self.conditions)
        return f"IF: {conds}\nTHEN: {self.outcome}"


@dataclass
class MatchExplanation:
    """Explicacion estructurada de una decision de match.

    Attributes:
        researcher_name: Nombre del investigador.
        candidate_name: Nombre del candidato.
        decision: "accepted" | "rejected" | "ambiguous"
        confidence: Score final de confianza.
        summary: Explicacion en texto natural (1-2 parrafos).
        key_factors: Factores principales de la decision.
        supporting_evidence: Evidencia que soporta la decision.
        weaknesses: Aspectos debiles o sin evidencia.
        convergence_note: Nota sobre la convergencia.
    """
    researcher_name: str = ""
    candidate_name: str = ""
    decision: str = "ambiguous"
    confidence: float = 0.0
    summary: str = ""
    key_factors: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    convergence_note: str = ""

    def to_dict(self) -> dict:
        return {
            "researcher_name": self.researcher_name,
            "candidate_name": self.candidate_name,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "key_factors": self.key_factors,
            "supporting_evidence": self.supporting_evidence,
            "weaknesses": self.weaknesses,
            "convergence_note": self.convergence_note,
        }

    def to_natural_text(self) -> str:
        """Genera explicacion en texto natural."""
        lines = []
        lines.append(self.summary)
        if self.key_factors:
            lines.append("")
            lines.append("Factores principales:")
            for f in self.key_factors:
                lines.append(f"  - {f}")
        if self.supporting_evidence:
            lines.append("")
            lines.append("Evidencia de soporte:")
            for e in self.supporting_evidence:
                lines.append(f"  - {e}")
        if self.weaknesses:
            lines.append("")
            lines.append("Limitaciones:")
            for w in self.weaknesses:
                lines.append(f"  - {w}")
        if self.convergence_note:
            lines.append("")
            lines.append(self.convergence_note)
        return "\n".join(lines)


# ─── Main Explainer Class ────────────────────────────────────────────────────

class MatchExplainer:
    """Motor de explicabilidad para decisiones del sistema.

    Genera explicaciones legibles y reglas interpretables
    a partir de las trazas de evidencia del refinement engine.

    Usage:
        explainer = MatchExplainer()
        explanation = explainer.explain_match(trace, "cand:openalex:A123")
        print(explanation.to_natural_text())

        rules = explainer.extract_rules([trace1, trace2, ...])
        for r in rules:
            print(r.to_text())
    """

    # Umbrales para clasificar scores
    HIGH_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.4
    LOW_THRESHOLD = 0.2

    def explain_match(
        self,
        trace: EvidenceTrace,
        candidate_id: Optional[str] = None,
    ) -> MatchExplanation:
        """Genera explicacion de por que se asigno un match.

        Si no se especifica candidate_id, usa el best_candidate.

        Args:
            trace: Traza de evidencia del refinamiento.
            candidate_id: ID del candidato (default: best_candidate).

        Returns:
            MatchExplanation con explicacion completa.
        """
        cid = candidate_id or trace.best_candidate_id
        if not cid:
            return MatchExplanation(
                researcher_name=trace.researcher_name,
                summary="No se encontraron candidatos para este investigador.",
                decision="rejected",
            )

        ct = trace.get_candidate_trace(cid)
        if not ct.trajectory:
            return MatchExplanation(
                researcher_name=trace.researcher_name,
                summary="No hay datos de trayectoria para este candidato.",
                decision="rejected",
            )

        final = ct.trajectory[-1]
        conf = final.confidence
        scores = final.scores

        # Classify decision
        if conf >= self.HIGH_THRESHOLD:
            decision = "accepted"
        elif conf >= self.MEDIUM_THRESHOLD:
            decision = "ambiguous"
        else:
            decision = "rejected"

        explanation = MatchExplanation(
            researcher_name=trace.researcher_name,
            candidate_name=ct.display_name,
            decision=decision,
            confidence=conf,
        )

        # Build summary
        explanation.summary = self._build_summary(
            trace, ct, decision, conf, scores
        )

        # Key factors: dimensions with highest scores
        sorted_dims = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )
        for dim, val in sorted_dims:
            label = _DIM_LABELS.get(dim, dim)
            if val >= self.HIGH_THRESHOLD:
                explanation.key_factors.append(
                    f"{label} es alta ({val:.2f})"
                )
            elif val >= self.MEDIUM_THRESHOLD:
                explanation.key_factors.append(
                    f"{label} es moderada ({val:.2f})"
                )

        # Supporting evidence from contributions
        explanation.supporting_evidence = self._extract_supporting_evidence(
            trace, cid
        )

        # Weaknesses: low dimensions
        for dim, val in sorted_dims:
            label = _DIM_LABELS.get(dim, dim)
            if val < self.LOW_THRESHOLD and dim != "coauthor":
                explanation.weaknesses.append(
                    f"{label} es baja ({val:.2f})"
                )

        # Convergence note
        explanation.convergence_note = self._convergence_note(trace, ct)

        return explanation

    def explain_rejection(
        self,
        trace: EvidenceTrace,
        candidate_id: str,
    ) -> MatchExplanation:
        """Explica por que un candidato fue rechazado.

        Args:
            trace: Traza de evidencia.
            candidate_id: ID del candidato rechazado.

        Returns:
            MatchExplanation con razones del rechazo.
        """
        ct = trace.get_candidate_trace(candidate_id)
        if not ct.trajectory:
            return MatchExplanation(
                researcher_name=trace.researcher_name,
                summary="No hay trayectoria para este candidato.",
                decision="rejected",
            )

        final = ct.trajectory[-1]
        best_ct = None
        if trace.best_candidate_id and trace.best_candidate_id != candidate_id:
            best_ct = trace.get_candidate_trace(trace.best_candidate_id)

        explanation = MatchExplanation(
            researcher_name=trace.researcher_name,
            candidate_name=ct.display_name,
            decision="rejected",
            confidence=final.confidence,
        )

        # Why rejected
        reasons = []
        for dim, val in final.scores.items():
            label = _DIM_LABELS.get(dim, dim)
            if val < self.MEDIUM_THRESHOLD:
                reasons.append(f"{label} insuficiente ({val:.2f})")

        if best_ct and best_ct.trajectory:
            best_final = best_ct.trajectory[-1]
            gap = best_final.confidence - final.confidence
            reasons.append(
                f"Existe un candidato con {gap:.2f} puntos mas de confianza "
                f"({best_ct.display_name})"
            )

        explanation.weaknesses = reasons
        explanation.summary = (
            f"El candidato {ct.display_name} fue rechazado para "
            f"{trace.researcher_name} con confidence {final.confidence:.2f}. "
            f"{'Las dimensiones debiles son: ' + ', '.join(reasons[:3]) + '.' if reasons else ''}"
        )

        return explanation

    def explain_convergence(
        self,
        trace: EvidenceTrace,
    ) -> str:
        """Genera explicacion de la convergencia del refinamiento.

        Returns:
            Texto explicando por que y como convergio.
        """
        n = trace.total_iterations
        conv = trace.converged

        if n == 0:
            return "No se ejecutaron iteraciones de refinamiento."

        lines = []
        if conv:
            lines.append(
                f"El sistema convergio en {n} iteracion{'es' if n > 1 else ''}."
            )
        else:
            lines.append(
                f"El sistema NO convergio tras {n} iteraciones "
                f"(limite alcanzado)."
            )

        # Analyze delta reduction
        deltas = [it.max_delta for it in trace.iterations]
        if len(deltas) >= 2:
            initial_delta = deltas[0]
            final_delta = deltas[-1]
            reduction = (
                (initial_delta - final_delta) / initial_delta * 100
                if initial_delta > 0 else 0
            )
            lines.append(
                f"El delta maximo se redujo de {initial_delta:.6f} a "
                f"{final_delta:.6f} ({reduction:.1f}% de reduccion)."
            )

        # Dominant dimensions across iterations
        dominant_dims = [
            it.dominant_dimension
            for it in trace.iterations
            if it.dominant_dimension
        ]
        if dominant_dims:
            from collections import Counter
            counts = Counter(dominant_dims)
            most_common = counts.most_common(2)
            labels = [
                f"{_DIM_LABELS.get(d, d)} ({c} iter)"
                for d, c in most_common
            ]
            lines.append(
                f"Dimensiones dominantes: {', '.join(labels)}."
            )

        return " ".join(lines)

    def extract_rules(
        self,
        traces: list[EvidenceTrace],
    ) -> list[InterpretableRule]:
        """Extrae reglas interpretables tipo decision tree.

        Analiza las trazas de multiples investigadores para encontrar
        patrones: IF score > threshold THEN outcome.

        Args:
            traces: Lista de trazas de evidencia.

        Returns:
            Lista de reglas interpretables.
        """
        rules = []

        # Gather final states of best candidates
        accepted = []   # high confidence matches
        ambiguous = []  # medium confidence
        rejected = []   # low confidence / no match

        for trace in traces:
            if not trace.best_candidate_id:
                continue
            ct = trace.get_candidate_trace(trace.best_candidate_id)
            if not ct.trajectory:
                continue
            final = ct.trajectory[-1]
            if final.confidence >= self.HIGH_THRESHOLD:
                accepted.append(final.scores)
            elif final.confidence >= self.MEDIUM_THRESHOLD:
                ambiguous.append(final.scores)
            else:
                rejected.append(final.scores)

        # Extract rules from accepted (high confidence)
        if accepted:
            rules.extend(
                self._extract_rules_from_group(
                    accepted, "high_confidence_match"
                )
            )

        # Extract rules from ambiguous
        if ambiguous:
            rules.extend(
                self._extract_rules_from_group(
                    ambiguous, "ambiguous_match"
                )
            )

        return rules

    # ── Private Helpers ───────────────────────────────────────────────

    def _build_summary(
        self,
        trace: EvidenceTrace,
        ct: CandidateTrace,
        decision: str,
        conf: float,
        scores: dict[str, float],
    ) -> str:
        """Construye parrafo de resumen."""
        if decision == "accepted":
            intro = (
                f"El ORCID/OpenAlex fue asignado a {trace.researcher_name} "
                f"con alta confianza ({conf:.1%}) porque:"
            )
        elif decision == "ambiguous":
            intro = (
                f"El candidato {ct.display_name} tiene confianza moderada "
                f"({conf:.1%}) para {trace.researcher_name}. "
                f"Se requiere mas evidencia para confirmar:"
            )
        else:
            intro = (
                f"El candidato {ct.display_name} tiene baja confianza "
                f"({conf:.1%}) para {trace.researcher_name}:"
            )

        reasons = []
        for dim, val in sorted(scores.items(), key=lambda x: -x[1]):
            label = _DIM_LABELS.get(dim, dim)
            if val >= self.HIGH_THRESHOLD:
                reasons.append(f"la {label} es alta ({val:.2f})")
            elif val >= self.MEDIUM_THRESHOLD:
                reasons.append(f"la {label} es moderada ({val:.2f})")

        if reasons:
            detail = ", ".join(reasons[:4])
            intro += f" {detail}."

        if ct.confidence_gain > 0.01:
            intro += (
                f" El confidence aumento {ct.confidence_gain:.2f} "
                f"durante {trace.total_iterations} iteraciones de refinamiento."
            )

        return intro

    def _extract_supporting_evidence(
        self,
        trace: EvidenceTrace,
        candidate_id: str,
    ) -> list[str]:
        """Extrae descripciones de evidencia de soporte."""
        evidence = []
        seen = set()

        for it in trace.iterations:
            for c in it.contributions:
                if c.was_applied and c.detail and c.detail not in seen:
                    evidence.append(c.detail)
                    seen.add(c.detail)

        return evidence[:10]  # Limit to 10

    def _convergence_note(
        self,
        trace: EvidenceTrace,
        ct: CandidateTrace,
    ) -> str:
        """Genera nota sobre convergencia."""
        if trace.converged:
            return (
                f"El refinamiento convergio en {trace.total_iterations} "
                f"iteraciones. El score semantico "
                f"{'aumento' if ct.confidence_gain > 0 else 'se mantuvo'} "
                f"consistentemente durante el proceso."
            )
        return (
            f"El refinamiento NO convergio tras {trace.total_iterations} "
            f"iteraciones. La identidad aun presenta incertidumbre."
        )

    def _extract_rules_from_group(
        self,
        score_lists: list[dict[str, float]],
        outcome: str,
    ) -> list[InterpretableRule]:
        """Extrae reglas de un grupo de scores."""
        if not score_lists:
            return []

        rules = []

        # Find common high dimensions
        for dim in DIMENSION_NAMES:
            values = [s.get(dim, 0.0) for s in score_lists]
            if not values:
                continue
            avg = sum(values) / len(values)
            min_val = min(values)

            # If most cases have this dimension high
            if min_val >= self.MEDIUM_THRESHOLD and avg >= 0.5:
                threshold = round(min_val - 0.05, 2)
                label = _DIM_LABELS.get(dim, dim)
                conditions = [
                    f"{dim}_score > {threshold:.2f}"
                ]
                # Find complementary dimensions
                for dim2 in DIMENSION_NAMES:
                    if dim2 == dim:
                        continue
                    values2 = [s.get(dim2, 0.0) for s in score_lists]
                    avg2 = sum(values2) / len(values2)
                    if avg2 >= self.MEDIUM_THRESHOLD:
                        threshold2 = round(min(values2) - 0.05, 2)
                        conditions.append(f"{dim2}_score > {threshold2:.2f}")

                if len(conditions) >= 2:
                    rule = InterpretableRule(
                        conditions=conditions[:4],
                        outcome=outcome,
                        confidence_range=(
                            min(sum(s.values()) / len(s) for s in score_lists),
                            max(sum(s.values()) / len(s) for s in score_lists),
                        ),
                        support=len(score_lists),
                    )
                    rules.append(rule)
                    break  # One rule per group is enough

        return rules
