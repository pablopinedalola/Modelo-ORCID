"""
main.py -- Pipeline Modelo-ORCID (Fases 1-4).

Fase 1: Carga CSV del SNII, normaliza nombres e instituciones.
Fase 2: Busca candidatos ORCID/OpenAlex, rankea por similitud.
Fase 3: Construye Knowledge Graph, ejecuta refinamiento iterativo.
Fase 4: Semantic matching + generacion de perfiles academicos.

Usage:
    python main.py
    python main.py --limit 3 --top 5
    python main.py --skip-orcid     # Solo OpenAlex
    python main.py --skip-semantic  # Sin embeddings
    python main.py --save           # Guardar JSON/HTML
    python main.py -v               # Verbose
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from config import RAW_DATA_DIR, OUTPUT_DIR, LOG_FORMAT, LOG_LEVEL, EVIDENCE_WEIGHTS
from src.loader import SNIILoader
from src.normalizer import NameNormalizer, InstitutionNormalizer
from src.retrieval import ORCIDClient, OpenAlexClient, CandidateRanker
from src.retrieval.candidate_ranker import CandidateRanker as _CR
from src.graph import AcademicKnowledgeGraph
from src.refinement import RefinementEngine
from src.refinement.refinement_engine import RefinementResult
from src.profiles import ProfileGenerator
from src.models.schemas import NormalizedRecord, Candidate, RetrievalResult

# Phase 2 weights for initial display
_P2W = _CR.PHASE2_WEIGHTS


def setup_logging(verbose: bool = False) -> None:
    """Configura logging global."""
    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)


# =====================================================================
# FASE 1: CARGA Y NORMALIZACION
# =====================================================================

def normalize_pipeline(csv_path: str | Path) -> list[NormalizedRecord]:
    """Carga y normaliza investigadores del SNII."""
    print("\n" + "=" * 70)
    print("  MODELO-ORCID -- Fase 1: Carga y Normalizacion")
    print("=" * 70)

    loader = SNIILoader()
    records = loader.load(csv_path)

    summary = loader.summary(records)
    print(f"\n  Total investigadores: {summary['total']}")
    print(f"  Por nivel: {summary['por_nivel']}")

    print(f"\n  Normalizando nombres e instituciones...")
    name_norm = NameNormalizer()
    inst_norm = InstitutionNormalizer()

    normalized: list[NormalizedRecord] = []
    for record in records:
        name_result = name_norm.normalize_record(record)
        inst_result = inst_norm.normalize_record_institution(record.institucion)
        nr = NormalizedRecord(
            original=record,
            normalized_name=name_result["normalized_name"],
            name_aliases=name_result["aliases"],
            normalized_institution=inst_result["normalized_institution"],
            institution_aliases=inst_result["institution_aliases"],
            name_tokens=name_result["tokens"],
            ror_id=inst_result["ror_id"],
        )
        normalized.append(nr)

    print(f"  {len(normalized)} investigadores normalizados\n")
    return normalized


# =====================================================================
# FASE 2: CANDIDATE RETRIEVAL
# =====================================================================

def retrieval_pipeline(
    normalized: list[NormalizedRecord],
    limit: int = 5,
    top_k: int = 5,
    skip_orcid: bool = False,
    skip_openalex: bool = False,
) -> list[RetrievalResult]:
    """Busca candidatos ORCID/OpenAlex y los rankea."""
    print("=" * 70)
    print("  MODELO-ORCID -- Fase 2: Candidate Retrieval")
    print("=" * 70)

    orcid_client = None if skip_orcid else ORCIDClient()
    openalex_client = None if skip_openalex else OpenAlexClient()
    ranker = CandidateRanker()

    results: list[RetrievalResult] = []
    to_process = normalized[:limit]

    print(f"\n  Buscando candidatos para {len(to_process)} investigadores...\n")

    for i, record in enumerate(to_process):
        start_time = time.time()
        name = record.original.full_name
        print(f"  [{i+1}/{len(to_process)}] {name}...", end=" ")

        all_candidates: list[Candidate] = []
        errors: list[str] = []
        orcid_count = 0
        openalex_count = 0

        if orcid_client:
            try:
                oc = orcid_client.search_researcher(record)
                orcid_count = len(oc)
                all_candidates.extend(oc)
            except Exception as e:
                errors.append(f"ORCID: {e}")

        if openalex_client:
            try:
                oa = openalex_client.search_authors(record)
                openalex_count = len(oa)
                all_candidates.extend(oa)
            except Exception as e:
                errors.append(f"OpenAlex: {e}")

        merged = ranker.merge_duplicates(all_candidates)
        ranked = ranker.rank(record, merged)
        top = ranked[:top_k]
        elapsed = time.time() - start_time

        result = RetrievalResult(
            snii_id=record.id,
            snii_name=name,
            candidates=top,
            orcid_candidates_count=orcid_count,
            openalex_candidates_count=openalex_count,
            search_time_seconds=elapsed,
            errors=errors,
        )
        results.append(result)

        best_score = top[0].evidence.confidence(_P2W) if top else 0.0
        print(
            f"{len(top)} cand "
            f"(O:{orcid_count} A:{openalex_count}) "
            f"best={best_score:.2f} [{elapsed:.1f}s]"
        )

    print()
    return results


# =====================================================================
# FASE 3+4: KNOWLEDGE GRAPH + REFINEMENT + SEMANTIC + PROFILES
# =====================================================================

def graph_refinement_pipeline(
    normalized: list[NormalizedRecord],
    retrieval_results: list[RetrievalResult],
    openalex_client: OpenAlexClient,
    skip_openalex: bool = False,
    skip_semantic: bool = False,
    save: bool = False,
) -> tuple[AcademicKnowledgeGraph, list[RefinementResult]]:
    """Builds graph, runs semantic matching + refinement, generates profiles.

    Args:
        normalized: Investigadores normalizados.
        retrieval_results: Resultados de busqueda.
        openalex_client: Cliente OpenAlex para enriquecer grafo.
        skip_openalex: Saltar enrichment OpenAlex.
        skip_semantic: Saltar embeddings semanticos.
        save: Guardar perfiles.

    Returns:
        Tupla (grafo, lista de RefinementResult).
    """
    print("=" * 70)
    print("  MODELO-ORCID -- Fase 3: Knowledge Graph + Refinement")
    print("=" * 70)

    graph = AcademicKnowledgeGraph()
    record_map = {r.id: r for r in normalized}

    # == PASO 3.1: Construir el grafo ==
    print("\n  Construyendo Knowledge Graph...")

    # Track paper titles for semantic matching
    paper_titles_by_candidate: dict[str, list[str]] = {}

    for result in retrieval_results:
        record = record_map.get(result.snii_id)
        if not record:
            continue

        researcher_id = graph.add_researcher(record)

        for candidate in result.candidates:
            cand_id = graph.add_candidate(candidate, researcher_id)

            if (not skip_openalex
                    and candidate.openalex_id
                    and openalex_client):
                try:
                    works = openalex_client.get_works(
                        candidate.openalex_id, limit=5
                    )
                    titles = []
                    for work in works:
                        graph.add_paper(work, cand_id)
                        t = work.get("title", "")
                        if t:
                            titles.append(t)
                    paper_titles_by_candidate[candidate.source_id] = titles
                except Exception:
                    pass

    stats = graph.stats()
    print(f"  Nodos: {stats['total_nodes']}  Aristas: {stats['total_edges']}")
    for ntype, count in sorted(stats['nodes_by_type'].items()):
        print(f"    {ntype:15s}: {count}")

    # == PASO 3.2: Semantic matcher ==
    semantic_matcher = None
    if not skip_semantic:
        print(f"\n{'=' * 70}")
        print(f"  MODELO-ORCID -- Fase 4: Semantic Matching")
        print(f"{'=' * 70}")
        print(f"\n  Cargando modelo de embeddings...")
        try:
            from src.semantic import SemanticMatcher
            semantic_matcher = SemanticMatcher()
            # Pre-warm: embed a test string to trigger model load
            semantic_matcher.engine.embed_text("test")
            print(f"  Modelo cargado exitosamente")
        except Exception as e:
            logger = logging.getLogger("pipeline")
            logger.warning(f"  No se pudo cargar modelo semantico: {e}")
            logger.warning(f"  Continuando sin semantic matching...")
            semantic_matcher = None

    # == PASO 3.3: Refinement ==
    print(f"\n  Ejecutando refinamiento iterativo" +
          (" + semantico" if semantic_matcher else "") + "...")
    print(f"  {'─' * 50}")

    engine = RefinementEngine(
        graph,
        semantic_matcher=semantic_matcher,
    )
    refinement_results: list[RefinementResult] = []

    for result in retrieval_results:
        record = record_map.get(result.snii_id)
        if not record:
            continue

        print(f"\n  {record.original.full_name}")

        ref_result = engine.refine(record, result.candidates)
        refinement_results.append(ref_result)

        _display_refinement_compact(ref_result, semantic_matcher is not None)

    # == PASO 4: Generar perfiles + guardar refinamiento ==
    if save:
        _generate_profiles(
            normalized, retrieval_results, refinement_results,
            record_map, paper_titles_by_candidate,
        )

        # Save refinement results JSON
        ref_data = []
        for ref_result in refinement_results:
            entry = {
                "researcher_id": ref_result.researcher_id,
                "researcher_name": ref_result.researcher_name,
                "iterations": ref_result.iterations,
                "converged": ref_result.converged,
                "best_candidate": None,
                "history": [
                    {"iteration": h.iteration, "max_delta": h.max_delta,
                     "converged": h.converged}
                    for h in ref_result.history
                ],
            }
            if ref_result.best_candidate:
                bc = ref_result.best_candidate
                ev = bc.evidence
                entry["best_candidate"] = {
                    "display_name": bc.display_name,
                    "orcid_id": bc.orcid_id,
                    "openalex_id": bc.openalex_id,
                    "evidence": {
                        "name_score": ev.name_score,
                        "institution_score": ev.institution_score,
                        "area_score": ev.area_score,
                        "publication_score": ev.publication_score,
                        "coauthor_score": ev.coauthor_score,
                        "temporal_score": ev.temporal_score,
                        "semantic_score": ev.semantic_score,
                        "confidence": round(ev.confidence(), 4),
                    },
                    "confidence": ev.confidence(),
                }
            ref_data.append(entry)

        ref_path = OUTPUT_DIR / "refinement_results.json"
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)

        # == Fase 7: Interpretability — Save traces & analysis ==
        _run_interpretability_analysis(refinement_results)

    return graph, refinement_results


def _run_interpretability_analysis(
    refinement_results: list[RefinementResult],
) -> None:
    """Ejecuta analisis de interpretabilidad y guarda resultados.

    Fase 7: Evidence Trace + Explainability + Ambiguity + Dynamics.
    """
    print(f"\n{'=' * 70}")
    print(f"  Fase 7: Interpretability Analysis")
    print(f"{'=' * 70}")

    from src.interpreter import (
        MatchExplainer,
        AmbiguityAnalyzer,
        DynamicsAnalyzer,
        StateVisualizer,
    )
    from src.math import MathematicalMapping

    traces_dir = OUTPUT_DIR / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Collect all traces
    traces = []
    for ref_result in refinement_results:
        if ref_result.trace:
            traces.append(ref_result.trace)
            # Save individual trace JSON
            trace_path = traces_dir / f"{ref_result.researcher_id.replace(':', '_')}_trace.json"
            ref_result.trace.save_json(trace_path)

    if not traces:
        print("  No hay trazas de evidencia disponibles.")
        return

    print(f"  {len(traces)} trazas de evidencia guardadas")

    # 1. Explainability
    explainer = MatchExplainer()
    explanations = []
    for trace in traces:
        expl = explainer.explain_match(trace)
        explanations.append(expl)

    # 2. Ambiguity Analysis
    amb_analyzer = AmbiguityAnalyzer()
    amb_reports = amb_analyzer.batch_analyze(traces)
    amb_stats = amb_analyzer.summary_stats(amb_reports)

    print(f"  Ambiguedad promedio: {amb_stats.get('avg_ambiguity', 0):.3f}")
    print(f"  Riesgo alto: {amb_stats.get('high_risk_count', 0)}, "
          f"medio: {amb_stats.get('medium_risk_count', 0)}, "
          f"bajo: {amb_stats.get('low_risk_count', 0)}")

    # 3. Dynamics Analysis
    dyn_analyzer = DynamicsAnalyzer()
    dyn_reports = dyn_analyzer.batch_analyze(traces)

    for dr in dyn_reports:
        print(f"  {dr.researcher_name[:35]:35s} "
              f"atractor={dr.attractor_type:20s} "
              f"estabilidad={dr.stability_score:.2f}")

    # 4. State Visualizations (save as JSON for frontend)
    visualizer = StateVisualizer(output_dir=str(figures_dir))
    for trace in traces:
        visualizer.save_visualization_data(trace)
    visualizer.save_landscape(traces)
    print(f"  Visualizaciones guardadas en {figures_dir}")

    # 5. Mathematical Verification
    mapping = MathematicalMapping()
    verifications = mapping.verify_properties(traces)
    print(f"\n  Verificacion de propiedades matematicas:")
    for v in verifications:
        status = "  ✓" if v.verified else "  ✗"
        print(f"    {status} {v.property_name}")

    # 6. Extract rules
    rules = explainer.extract_rules(traces)
    if rules:
        print(f"\n  Reglas interpretables extraidas: {len(rules)}")
        for r in rules[:3]:
            print(f"    {r.to_text().replace(chr(10), ' | ')}")

    # 7. Generate research report
    _generate_research_report(
        traces, explanations, amb_reports, amb_stats,
        dyn_reports, verifications, rules, mapping,
        reports_dir / "research_analysis.md",
    )
    print(f"\n  Reporte academico: {reports_dir / 'research_analysis.md'}")

    # 8. Save analysis summary JSON
    analysis_data = {
        "ambiguity": {
            "summary": amb_stats,
            "reports": [r.to_dict() for r in amb_reports],
        },
        "dynamics": [r.to_dict() for r in dyn_reports],
        "explanations": [e.to_dict() for e in explanations],
        "verifications": [v.to_dict() for v in verifications],
        "rules": [{"conditions": r.conditions, "outcome": r.outcome}
                  for r in rules],
    }
    analysis_path = OUTPUT_DIR / "interpretability_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)


def _generate_research_report(
    traces, explanations, amb_reports, amb_stats,
    dyn_reports, verifications, rules, mapping,
    output_path: Path,
) -> None:
    """Genera reporte de investigacion en Markdown."""
    lines = []
    lines.append("# Modelo-ORCID: Análisis de Interpretabilidad")
    lines.append("")
    lines.append(f"> Generado automáticamente — {len(traces)} investigadores analizados")
    lines.append("")

    # Mathematical Mapping
    lines.append("## 1. Mapeo Matemático")
    lines.append("")
    lines.append(mapping.export_markdown_table())
    lines.append("")

    # Verification
    lines.append("## 2. Verificación de Propiedades")
    lines.append("")
    lines.append(mapping.export_verification_markdown(verifications))
    lines.append("")

    # Ambiguity
    lines.append("## 3. Análisis de Ambigüedad")
    lines.append("")
    lines.append(f"- Ambigüedad promedio: {amb_stats.get('avg_ambiguity', 0):.4f}")
    lines.append(f"- Entropía promedio: {amb_stats.get('avg_entropy', 0):.4f}")
    lines.append(f"- Gap de confidence promedio: {amb_stats.get('avg_confidence_gap', 0):.4f}")
    lines.append("")
    lines.append("| Investigador | Ambigüedad | Entropía | Gap | Riesgo |")
    lines.append("|---|---|---|---|---|")
    for ar in amb_reports:
        lines.append(
            f"| {ar.researcher_name[:30]} | {ar.ambiguity_score:.3f} | "
            f"{ar.entropy:.3f} | {ar.confidence_gap:.3f} | {ar.risk_level} |"
        )
    lines.append("")

    # Dynamics
    lines.append("## 4. Dinámica del Refinamiento")
    lines.append("")
    lines.append("| Investigador | Iteraciones | Convergió | Atractor | Estabilidad | Dim. Dominante |")
    lines.append("|---|---|---|---|---|---|")
    for dr in dyn_reports:
        lines.append(
            f"| {dr.researcher_name[:25]} | {dr.convergence.total_iterations} | "
            f"{'Sí' if dr.convergence.converged else 'No'} | "
            f"{dr.attractor_type} | {dr.stability_score:.2f} | "
            f"{dr.dominant_dimension} |"
        )
    lines.append("")

    # Explanations
    lines.append("## 5. Explicaciones de Decisiones")
    lines.append("")
    for expl in explanations:
        lines.append(f"### {expl.researcher_name}")
        lines.append(f"**Decisión**: {expl.decision} ({expl.confidence:.1%})")
        lines.append("")
        lines.append(expl.to_natural_text())
        lines.append("")

    # Rules
    if rules:
        lines.append("## 6. Reglas Interpretables")
        lines.append("")
        lines.append("```")
        for r in rules:
            lines.append(r.to_text())
            lines.append("")
        lines.append("```")
        lines.append("")

    # LaTeX table
    lines.append("## 7. Tabla LaTeX para Tesis")
    lines.append("")
    lines.append("```latex")
    lines.append(mapping.export_latex_table())
    lines.append("```")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _display_refinement_compact(
    result: RefinementResult,
    has_semantic: bool = False,
) -> None:
    """Muestra refinamiento de forma compacta."""
    if not result.history:
        print("    Sin candidatos")
        return

    for log in result.history:
        scores_str = " ".join(
            f"{s:.3f}" for s in sorted(log.scores.values(), reverse=True)[:3]
        )
        status = " [CONV]" if log.converged else ""
        print(f"    Iter {log.iteration}: top=[{scores_str}] d={log.max_delta:.5f}{status}")

    if result.best_candidate:
        bc = result.best_candidate
        conf = bc.evidence.confidence()
        ev = bc.evidence

        print(f"    Mejor: {bc.display_name}")
        print(f"    ORCID: {bc.orcid_id or '---'}")

        dims = (f"N={ev.name_score:.2f} I={ev.institution_score:.2f} "
                f"A={ev.area_score:.2f} P={ev.publication_score:.2f} "
                f"C={ev.coauthor_score:.2f} T={ev.temporal_score:.2f}")
        if has_semantic:
            dims += f" S={ev.semantic_score:.2f}"
        print(f"    Dims:  {dims}")
        print(f"    Conf:  {conf:.4f} ({result.iterations} iter, "
              f"{'converged' if result.converged else 'NOT converged'})")


def _generate_profiles(
    normalized: list[NormalizedRecord],
    retrieval_results: list[RetrievalResult],
    refinement_results: list[RefinementResult],
    record_map: dict[str, NormalizedRecord],
    paper_titles_by_candidate: dict[str, list[str]],
) -> None:
    """Genera perfiles HTML/JSON para cada investigador."""
    print(f"\n{'=' * 70}")
    print(f"  Generando perfiles academicos...")
    print(f"{'=' * 70}")

    gen = ProfileGenerator()

    for ref_result in refinement_results:
        researcher_id = ref_result.researcher_id
        snii_id = researcher_id.replace("snii:", "")
        record = record_map.get(snii_id)
        if not record:
            continue

        bc = ref_result.best_candidate
        confidence = bc.evidence.confidence() if bc else 0.0

        # Gather papers
        papers = []
        if bc and bc.source_id in paper_titles_by_candidate:
            for title in paper_titles_by_candidate[bc.source_id]:
                papers.append({"title": title})

        # Gather papers from graph with full data
        if bc:
            from src.graph import AcademicKnowledgeGraph
            cand_graph_id = f"cand:{bc.source.value}:{bc.source_id}"

        profile = gen.generate_profile(
            record=record,
            candidate=bc,
            confidence=confidence,
            papers=papers,
            semantic_score=bc.evidence.semantic_score if bc else 0.0,
        )

        paths = gen.save_profile(profile, formats=("html", "json"))
        print(f"  {record.original.full_name[:35]:35s} -> {paths[0].name}")


# =====================================================================
# RESUMEN FINAL
# =====================================================================

def display_final_summary(
    refinement_results: list[RefinementResult],
    graph: AcademicKnowledgeGraph,
    has_semantic: bool = False,
) -> None:
    """Muestra resumen de todo el pipeline."""
    print(f"\n{'=' * 70}")
    print(f"  RESUMEN FINAL")
    print(f"{'=' * 70}")

    stats = graph.stats()
    print(f"\n  Knowledge Graph: {stats['total_nodes']} nodos, {stats['total_edges']} aristas")

    total = len(refinement_results)
    converged = sum(1 for r in refinement_results if r.converged)
    avg_iter = sum(r.iterations for r in refinement_results) / total if total else 0

    with_orcid = sum(
        1 for r in refinement_results
        if r.best_candidate and r.best_candidate.orcid_id
    )
    high_conf = sum(
        1 for r in refinement_results
        if r.best_candidate and r.best_candidate.evidence.confidence() > 0.5
    )

    print(f"\n  Investigadores:   {total}")
    print(f"  Convergieron:     {converged}/{total}")
    print(f"  Iter promedio:    {avg_iter:.1f}")
    print(f"  Con ORCID:        {with_orcid}/{total}")
    print(f"  Confidence > 0.5: {high_conf}/{total}")
    if has_semantic:
        avg_sem = sum(
            r.best_candidate.evidence.semantic_score
            for r in refinement_results if r.best_candidate
        ) / total if total else 0
        print(f"  Semantic avg:     {avg_sem:.3f}")

    print(f"\n  Ranking final:")
    sorted_results = sorted(
        [r for r in refinement_results if r.best_candidate],
        key=lambda r: r.best_candidate.evidence.confidence(),
        reverse=True,
    )
    for r in sorted_results:
        bc = r.best_candidate
        conf = bc.evidence.confidence()
        bar_len = int(conf * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        orcid = bc.orcid_id or "---"
        sem = f" S={bc.evidence.semantic_score:.2f}" if has_semantic else ""
        print(
            f"    [{bar}] {conf:.3f}{sem} | "
            f"{r.researcher_name[:28]:28s} | "
            f"{bc.display_name[:22]:22s} | {orcid}"
        )


# =====================================================================
# ENTRY POINT
# =====================================================================

def main() -> None:
    """Entry point — pipeline completo Fases 1-4."""
    parser = argparse.ArgumentParser(
        description="Modelo-ORCID: Pipeline de desambiguacion academica"
    )
    parser.add_argument(
        "--csv", type=str,
        default=str(RAW_DATA_DIR / "snii_sample.csv"),
        help="Ruta al CSV/Excel del SNII",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=3,
        help="Investigadores a procesar (default: 3)",
    )
    parser.add_argument(
        "--top", "-k", type=int, default=5,
        help="Top K candidatos por investigador (default: 5)",
    )
    parser.add_argument(
        "--skip-orcid", action="store_true",
        help="Saltar busqueda ORCID",
    )
    parser.add_argument(
        "--skip-openalex", action="store_true",
        help="Saltar busqueda OpenAlex",
    )
    parser.add_argument(
        "--skip-semantic", action="store_true",
        help="Saltar semantic matching (sin embeddings)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detallado",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Guardar resultados JSON + grafo + perfiles HTML",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    pipeline_start = time.time()

    try:
        # ── Fase 1 ──
        normalized = normalize_pipeline(args.csv)

        # ── Fase 2 ──
        retrieval_results = retrieval_pipeline(
            normalized,
            limit=args.limit,
            top_k=args.top,
            skip_orcid=args.skip_orcid,
            skip_openalex=args.skip_openalex,
        )

        # ── Fase 3 + 4 ──
        openalex_client = None if args.skip_openalex else OpenAlexClient()
        graph, refinement_results = graph_refinement_pipeline(
            normalized,
            retrieval_results,
            openalex_client=openalex_client,
            skip_openalex=args.skip_openalex,
            skip_semantic=args.skip_semantic,
            save=args.save,
        )

        # ── Resumen final ──
        has_semantic = not args.skip_semantic
        display_final_summary(refinement_results, graph, has_semantic)

        # ── Guardar grafo ──
        if args.save:
            graph_path = OUTPUT_DIR / "knowledge_graph.json"
            graph.save_json(str(graph_path))
            print(f"\n  Grafo: {graph_path}")

        elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 70}")
        print(f"  Pipeline completo en {elapsed:.1f}s")
        print(f"  Fases ejecutadas: 1 (Normalizacion) + 2 (Retrieval) "
              f"+ 3 (Graph/Refinement)"
              + (" + 4 (Semantic)" if has_semantic else ""))
        print(f"{'=' * 70}\n")

    except KeyboardInterrupt:
        print("\n\n  Interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        logging.getLogger("pipeline").error(
            f"Error en el pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
