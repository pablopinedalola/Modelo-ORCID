"""
state_visualizer.py -- Visualizacion de la evolucion del estado.

Genera visualizaciones de:
    d_0(v) -> d_1(v) -> d_2(v) -> ... -> d*(v)

Produce figuras guardadas como PNG usando matplotlib:
- Confidence trajectory charts
- Dimension evolution heatmaps
- Delta convergence plots
- Ambiguity landscapes

Usage:
    viz = StateVisualizer(output_dir="data/reports/figures")
    viz.plot_confidence_trajectory(trace, candidate_id)
    viz.plot_dimension_evolution(trace, candidate_id)
    viz.plot_convergence_landscape(traces)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.interpreter.evidence_trace import (
    EvidenceTrace,
    DIMENSION_NAMES,
)
from src.interpreter.ambiguity_analysis import AmbiguityAnalyzer, AmbiguityReport
from src.interpreter.dynamics import DynamicsAnalyzer, DynamicsReport


class StateVisualizer:
    """Generador de visualizaciones del estado del refinamiento.

    Genera datos de visualizacion en formato JSON que pueden ser
    renderizados con Chart.js en el frontend o con matplotlib
    para exportacion a PDF/LaTeX.

    Attributes:
        output_dir: Directorio donde guardar figuras.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/reports/figures",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Confidence Trajectory ─────────────────────────────────────────

    def confidence_trajectory_data(
        self,
        trace: EvidenceTrace,
        candidate_id: Optional[str] = None,
    ) -> dict:
        """Genera datos de trayectoria de confidence.

        Si no se especifica candidate_id, genera para todos.

        Returns:
            Dict con structure para Chart.js:
            {labels: [0,1,2,...], datasets: [{label, data}]}
        """
        traces = trace.get_all_candidate_traces()
        if candidate_id:
            traces = {candidate_id: traces.get(candidate_id)}

        labels = list(range(trace.total_iterations + 1))  # 0..N
        datasets = []

        # Color palette
        colors = [
            "#818cf8", "#4ade80", "#fb923c", "#f87171",
            "#22d3ee", "#c084fc", "#fbbf24",
        ]

        for i, (cid, ct) in enumerate(traces.items()):
            if ct is None or not ct.trajectory:
                continue
            data = [round(s.confidence, 4) for s in ct.trajectory]
            # Pad with last value if needed
            while len(data) < len(labels):
                data.append(data[-1] if data else 0)

            is_best = cid == trace.best_candidate_id
            datasets.append({
                "label": ct.display_name[:30] or cid[:30],
                "data": data,
                "borderColor": colors[i % len(colors)],
                "borderWidth": 3 if is_best else 1.5,
                "fill": False,
                "tension": 0.3,
                "pointRadius": 4 if is_best else 2,
            })

        return {
            "type": "confidence_trajectory",
            "researcher": trace.researcher_name,
            "labels": labels,
            "datasets": datasets,
        }

    # ── Dimension Evolution ───────────────────────────────────────────

    def dimension_evolution_data(
        self,
        trace: EvidenceTrace,
        candidate_id: Optional[str] = None,
    ) -> dict:
        """Genera datos de evolucion por dimension para un candidato.

        Returns:
            Dict con datos de radar/bar chart por iteracion.
        """
        cid = candidate_id or trace.best_candidate_id
        if not cid:
            return {"type": "dimension_evolution", "datasets": []}

        ct = trace.get_candidate_trace(cid)
        if not ct.trajectory:
            return {"type": "dimension_evolution", "datasets": []}

        labels = DIMENSION_NAMES
        datasets = []

        dim_labels = {
            "name": "Nombre", "institution": "Institución",
            "area": "Área", "publication": "Publicaciones",
            "coauthor": "Coautores", "temporal": "Temporal",
            "semantic": "Semántico",
        }

        colors_alpha = [
            "rgba(129,140,248,{a})", "rgba(74,222,128,{a})",
            "rgba(251,146,60,{a})", "rgba(248,113,113,{a})",
        ]

        for i, snap in enumerate(ct.trajectory):
            data = [round(snap.scores.get(d, 0.0), 4) for d in labels]
            alpha = 0.3 + (0.7 * i / max(len(ct.trajectory) - 1, 1))
            color_template = colors_alpha[i % len(colors_alpha)]

            datasets.append({
                "label": f"Iteración {i}",
                "data": data,
                "backgroundColor": color_template.format(a=f"{alpha:.2f}"),
                "borderColor": color_template.format(a="1"),
                "borderWidth": 2 if i == len(ct.trajectory) - 1 else 1,
            })

        return {
            "type": "dimension_evolution",
            "researcher": trace.researcher_name,
            "candidate": ct.display_name,
            "labels": [dim_labels.get(d, d) for d in labels],
            "raw_labels": labels,
            "datasets": datasets,
        }

    # ── Delta Convergence ─────────────────────────────────────────────

    def convergence_data(
        self,
        trace: EvidenceTrace,
    ) -> dict:
        """Genera datos del plot de convergencia (delta vs iteracion).

        Returns:
            Dict con datos para line chart.
        """
        deltas = [it.max_delta for it in trace.iterations]
        labels = list(range(1, len(deltas) + 1))

        return {
            "type": "convergence",
            "researcher": trace.researcher_name,
            "labels": labels,
            "datasets": [{
                "label": "Max Delta (Δ)",
                "data": [round(d, 6) for d in deltas],
                "borderColor": "#818cf8",
                "backgroundColor": "rgba(129,140,248,0.1)",
                "fill": True,
                "tension": 0.3,
                "borderWidth": 2,
            }],
            "epsilon": 0.01,  # Convergence threshold
        }

    # ── Ambiguity Landscape ───────────────────────────────────────────

    def ambiguity_landscape_data(
        self,
        traces: list[EvidenceTrace],
    ) -> dict:
        """Genera datos del landscape de ambiguedad para multiples casos.

        Returns:
            Dict con datos de scatter/bubble chart.
        """
        analyzer = AmbiguityAnalyzer()
        points = []

        for trace in traces:
            report = analyzer.analyze(trace)
            # Get best confidence
            best_conf = 0.0
            if trace.best_candidate_id:
                ct = trace.get_candidate_trace(trace.best_candidate_id)
                if ct.trajectory:
                    best_conf = ct.trajectory[-1].confidence

            points.append({
                "x": round(report.ambiguity_score, 4),
                "y": round(best_conf, 4),
                "r": max(5, int(report.entropy * 15)),
                "label": trace.researcher_name[:25],
                "risk": report.risk_level,
            })

        color_map = {
            "low": "#4ade80",
            "medium": "#fb923c",
            "high": "#f87171",
        }

        datasets = {}
        for p in points:
            risk = p["risk"]
            if risk not in datasets:
                datasets[risk] = {
                    "label": f"Riesgo {risk}",
                    "data": [],
                    "backgroundColor": color_map.get(risk, "#999") + "80",
                    "borderColor": color_map.get(risk, "#999"),
                }
            datasets[risk]["data"].append(p)

        return {
            "type": "ambiguity_landscape",
            "datasets": list(datasets.values()),
        }

    # ── Evidence Propagation ──────────────────────────────────────────

    def propagation_data(
        self,
        trace: EvidenceTrace,
    ) -> dict:
        """Genera datos de propagacion de evidencia.

        Returns:
            Dict con datos para Sankey-like visualization.
        """
        analyzer = DynamicsAnalyzer()
        report = analyzer.analyze(trace)

        # Build evidence flow: neighbor_type -> dimension -> count
        flows = []
        for it in trace.iterations:
            for c in it.contributions:
                if c.was_applied:
                    flows.append({
                        "source": c.neighbor_type,
                        "target": c.dimension,
                        "value": round(c.delta, 4),
                    })

        # Aggregate flows
        agg: dict[str, float] = {}
        for f in flows:
            key = f"{f['source']}→{f['target']}"
            agg[key] = agg.get(key, 0.0) + f["value"]

        return {
            "type": "propagation",
            "researcher": trace.researcher_name,
            "flows": [
                {
                    "source": k.split("→")[0],
                    "target": k.split("→")[1],
                    "value": round(v, 4),
                }
                for k, v in sorted(agg.items(), key=lambda x: -x[1])
            ],
            "propagation_stats": report.propagation.to_dict(),
        }

    # ── Combined Report Data ──────────────────────────────────────────

    def full_visualization_data(
        self,
        trace: EvidenceTrace,
    ) -> dict:
        """Genera todos los datos de visualizacion para un caso.

        Returns:
            Dict con todos los charts combinados.
        """
        return {
            "researcher_id": trace.researcher_id,
            "researcher_name": trace.researcher_name,
            "confidence_trajectory": self.confidence_trajectory_data(trace),
            "dimension_evolution": self.dimension_evolution_data(trace),
            "convergence": self.convergence_data(trace),
            "propagation": self.propagation_data(trace),
        }

    def save_visualization_data(
        self,
        trace: EvidenceTrace,
        filename: Optional[str] = None,
    ) -> Path:
        """Guarda datos de visualizacion como JSON.

        Args:
            trace: Traza de evidencia.
            filename: Nombre del archivo (default: researcher_id).

        Returns:
            Path del archivo guardado.
        """
        data = self.full_visualization_data(trace)
        name = filename or trace.researcher_id.replace(":", "_")
        path = self.output_dir / f"{name}_viz.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return path

    def save_landscape(
        self,
        traces: list[EvidenceTrace],
        filename: str = "ambiguity_landscape.json",
    ) -> Path:
        """Guarda landscape de ambiguedad."""
        data = self.ambiguity_landscape_data(traces)
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
