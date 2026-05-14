#!/usr/bin/env python3
"""
evaluate_retrieval.py — Evaluación comparativa de retrieval.

Evalúa el rendimiento del retrieval semántico vs el retrieval híbrido.
Calcula métricas como similitud, rank changes, y overlap, generando un reporte
en Markdown con los resultados.

Uso:
    python scripts/evaluate_retrieval.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

# Configurar path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.hybrid_retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluator")

# Queries de prueba
TEST_QUERIES = [
    # Consultas semánticas (conceptos)
    "física de partículas",
    "matemáticas aplicadas",
    "ingeniería mecatrónica",
    "redes complejas",
    "óptica cuántica",
    "relaciones internacionales",
    # Consultas de metadata (institución/área explícitas)
    "UNAM física",
    "investigadores IPN",
    "biología UAM",
    "química teórica",
    "ciencias sociales",
    # Queries difíciles/ruidosos
    "experto en grafos y redes",
    "alguien que estudie astrofísica observacional",
    "ingeniería de software UDG",
]

def evaluate():
    logger.info("=" * 60)
    logger.info("📊 EVALUACIÓN DE RETRIEVAL: Semántico vs Híbrido")
    logger.info("=" * 60)

    # 1. Cargar retrievers
    logger.info("📥 Cargando motor híbrido...")
    retriever = HybridRetriever()
    if not retriever.load():
        logger.error("❌ No se pudo cargar el retriever. Asegúrate de haber ejecutado build_vector_index.py")
        sys.exit(1)

    logger.info(f"   ✓ Configuración híbrida: {retriever.semantic_weight:.2f} semántico, {retriever.lexical_weight:.2f} léxico")
    
    results_log = []
    
    # Métricas
    metrics = {
        "queries": 0,
        "avg_semantic_score": 0.0,
        "avg_hybrid_score": 0.0,
        "avg_rank_change": 0.0,
        "top_1_matches": 0,
    }

    print("\n🔍 Ejecutando queries...")
    print("-" * 80)
    
    for query in TEST_QUERIES:
        metrics["queries"] += 1
        
        # 1. Retrieval puramente Semántico (usando pesos 1.0, 0.0)
        retriever.semantic_weight = 1.0
        retriever.lexical_weight = 0.0
        sem_res = retriever.search(query, top_k=5, include_explanation=False)
        sem_top_id = sem_res[0]["id"] if sem_res else None
        sem_top_score = sem_res[0]["score"] if sem_res else 0.0
        metrics["avg_semantic_score"] += sem_top_score

        # 2. Retrieval Híbrido (restaurando pesos por defecto)
        from config import HYBRID_SEMANTIC_WEIGHT, HYBRID_LEXICAL_WEIGHT
        retriever.semantic_weight = HYBRID_SEMANTIC_WEIGHT
        retriever.lexical_weight = HYBRID_LEXICAL_WEIGHT
        hyb_res = retriever.search(query, top_k=5, include_explanation=True)
        hyb_top_id = hyb_res[0]["id"] if hyb_res else None
        hyb_top_score = hyb_res[0]["score"] if hyb_res else 0.0
        metrics["avg_hybrid_score"] += hyb_top_score

        # 3. Comparación
        overlap = len(set(r["id"] for r in sem_res) & set(r["id"] for r in hyb_res))
        
        if sem_top_id == hyb_top_id:
            metrics["top_1_matches"] += 1
            change = "Igual"
        else:
            change = "Cambiado"

        # Log
        print(f"Q: \"{query}\"")
        print(f"   Semántico Top 1: {sem_res[0]['nombre_completo'] if sem_res else 'Ninguno'} ({sem_top_score:.3f})")
        print(f"   Híbrido   Top 1: {hyb_res[0]['nombre_completo'] if hyb_res else 'Ninguno'} ({hyb_top_score:.3f})")
        print(f"   Overlap top-5: {overlap}/5 | Status: {change}")
        if hyb_res:
            print(f"   Explicación: {hyb_res[0]['explanation']}")
        print("-" * 80)

        results_log.append({
            "query": query,
            "sem_top_name": sem_res[0]['nombre_completo'] if sem_res else "",
            "sem_top_score": round(sem_top_score, 3),
            "hyb_top_name": hyb_res[0]['nombre_completo'] if hyb_res else "",
            "hyb_top_score": round(hyb_top_score, 3),
            "overlap_top5": overlap,
            "status": change
        })

    # Calcular promedios
    q_count = metrics["queries"]
    metrics["avg_semantic_score"] /= q_count
    metrics["avg_hybrid_score"] /= q_count

    # Generar reporte CSV
    out_dir = PROJECT_ROOT / "data" / "outputs"
    out_dir.mkdir(exist_ok=True)
    
    csv_path = out_dir / "retrieval_evaluation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)

    logger.info("")
    logger.info("=" * 60)
    logger.info("📈 RESULTADOS DE LA EVALUACIÓN")
    logger.info("=" * 60)
    logger.info(f"   Queries evaluados:    {q_count}")
    logger.info(f"   Score Promedio (Sem): {metrics['avg_semantic_score']:.3f}")
    logger.info(f"   Score Promedio (Hyb): {metrics['avg_hybrid_score']:.3f}")
    logger.info(f"   Top-1 Idénticos:      {metrics['top_1_matches']}/{q_count} ({metrics['top_1_matches']/q_count:.0%})")
    logger.info(f"   Reporte guardado en:  {csv_path.relative_to(PROJECT_ROOT)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    evaluate()
