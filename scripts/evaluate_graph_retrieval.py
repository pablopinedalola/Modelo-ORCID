#!/usr/bin/env python3
"""
evaluate_graph_retrieval.py — Evaluación comparativa Híbrido vs Grafos.

Compara el HybridRetriever (baseline) con el GraphAwareRetriever
para medir el impacto de la propagación de evidencia.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

# Configurar path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.hybrid_retriever import HybridRetriever
from src.graph.graph_enrichment import AcademicGraphBuilder
from src.rag.graph_aware_retriever import GraphAwareRetriever
from config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("evaluator")

TEST_QUERIES = [
    "física de partículas UNAM",
    "matemáticas",
    "redes complejas",
    "ingeniería UDG",
    "ciencias sociales y humanidades",
    "experto en grafos",
    "investigadores IPN",
    "óptica",
]

def evaluate():
    logger.info("=" * 60)
    logger.info("📊 EVALUACIÓN: HÍBRIDO vs GRAPH-AWARE")
    logger.info("=" * 60)

    logger.info("📥 Cargando retrievers...")
    
    hybrid = HybridRetriever()
    hybrid.load()
    
    profiles_path = PROCESSED_DATA_DIR / "snii_profiles.json"
    with open(profiles_path) as f:
        profiles = json.load(f)
        
    builder = AcademicGraphBuilder()
    builder.build_from_profiles(profiles)
    
    graph_retriever = GraphAwareRetriever(hybrid, builder)
    graph_retriever.load()
    
    results_log = []
    metrics = {
        "queries": 0,
        "avg_hyb_score": 0.0,
        "avg_graph_score": 0.0,
        "graph_boosted_queries": 0,
    }

    print("\n🔍 Ejecutando queries...")
    print("-" * 80)
    
    for query in TEST_QUERIES:
        metrics["queries"] += 1
        
        # 1. Híbrido
        hyb_res = hybrid.search(query, top_k=5, include_explanation=False)
        hyb_top_id = hyb_res[0]["id"] if hyb_res else None
        hyb_top_score = hyb_res[0]["score"] if hyb_res else 0.0
        metrics["avg_hyb_score"] += hyb_top_score

        # 2. Graph-Aware
        gr_res = graph_retriever.search(query, top_k=5)
        gr_top_id = gr_res[0]["id"] if gr_res else None
        gr_top_score = gr_res[0]["score"] if gr_res else 0.0
        metrics["avg_graph_score"] += gr_top_score

        # Boost
        boost = gr_top_score - hyb_top_score
        if boost > 0.001:
            metrics["graph_boosted_queries"] += 1
            change = "↗️  Boosted"
        elif gr_top_id != hyb_top_id:
            change = "🔀 Rank Changed"
        else:
            change = "➖ Igual"

        # Log
        print(f"Q: \"{query}\"")
        print(f"   Híbrido : {hyb_res[0]['nombre_completo'] if hyb_res else 'Ninguno'} ({hyb_top_score:.3f})")
        print(f"   Grafo   : {gr_res[0]['nombre_completo'] if gr_res else 'Ninguno'} ({gr_top_score:.3f})")
        print(f"   Status  : {change} (+{boost:.3f})")
        if gr_res:
            print(f"   Explicación: {gr_res[0].get('explanation', '')}")
        print("-" * 80)

        results_log.append({
            "query": query,
            "hyb_top_name": hyb_res[0]['nombre_completo'] if hyb_res else "",
            "hyb_top_score": round(hyb_top_score, 3),
            "graph_top_name": gr_res[0]['nombre_completo'] if gr_res else "",
            "graph_top_score": round(gr_top_score, 3),
            "boost": round(boost, 3),
            "status": change
        })

    q_count = metrics["queries"]
    metrics["avg_hyb_score"] /= q_count
    metrics["avg_graph_score"] /= q_count

    # Guardar
    out_dir = PROJECT_ROOT / "data" / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "graph_retrieval_evaluation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)

    logger.info("")
    logger.info("=" * 60)
    logger.info("📈 RESULTADOS DE LA EVALUACIÓN GRAFOS")
    logger.info("=" * 60)
    logger.info(f"   Queries evaluados:    {q_count}")
    logger.info(f"   Score Promedio (Hyb): {metrics['avg_hyb_score']:.3f}")
    logger.info(f"   Score Promedio (Gra): {metrics['avg_graph_score']:.3f}")
    logger.info(f"   Queries con Boost:    {metrics['graph_boosted_queries']}/{q_count} ({metrics['graph_boosted_queries']/q_count:.0%})")
    logger.info(f"   Reporte guardado en:  {csv_path.relative_to(PROJECT_ROOT)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    evaluate()
