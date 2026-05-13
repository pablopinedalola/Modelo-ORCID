#!/usr/bin/env python3
"""
evaluate_rag_quality.py — Evalúa la calidad del Academic RAG y Retrieval Híbrido.

Prueba diversas queries académicas complejas para asegurar que:
- El Query Understanding expanda sinónimos.
- El modelo multilingüe entienda queries cruzados (español <-> inglés).
- El sistema recupere no solo por keyword sino por topic/concept overlap.
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from src.rag.multi_vector_retriever import MultiVectorRetriever
from src.rag.query_understanding import QueryAnalyzer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval")

def run_evaluation():
    logger.info("=" * 80)
    logger.info("🧠 ACADEMIC RAG - EVALUACIÓN DE RETRIEVAL")
    logger.info("=" * 80)

    # 1. Test Query Understanding Isolated
    analyzer = QueryAnalyzer()
    test_queries = [
        "teoria de graficas",
        "inteligencia artificial para medicina",
        "investigadores que trabajen en superconductividad",
        "papers sobre covid y redes neuronales",
        "quantum optics"
    ]
    
    logger.info("\n1. 🔍 PRUEBA DE QUERY UNDERSTANDING & EXPANSION")
    logger.info("-" * 80)
    for q in test_queries:
        res = analyzer.analyze(q)
        logger.info(f"Query original: '{q}'")
        logger.info(f"  Intención: {res['intent']}")
        logger.info(f"  Términos detectados: {', '.join(res['expanded_terms'])}")
        logger.info(f"  Temas detectados: {', '.join(res['detected_topics']) if res['detected_topics'] else 'Ninguno'}")
        logger.info("")

    # 2. Test Multi-Vector Retrieval
    store_dir = DATA_DIR / "vector_store"
    retriever = MultiVectorRetriever(store_dir=store_dir)
    
    logger.info("\n2. 🚀 PRUEBA DE MULTI-VECTOR RETRIEVAL")
    logger.info("-" * 80)
    
    if not retriever.load():
        logger.warning("No se pudieron cargar los índices. Debes correr primero 'python3 scripts/build_vector_index.py'")
        return

    for q in test_queries:
        logger.info(f"\nBusqueda RAG: '{q}'")
        results = retriever.search(q, top_k=3)
        
        if not results:
            logger.info("  (Sin resultados)")
            continue
            
        for i, r in enumerate(results, 1):
            name = r.get('display_name', 'Desconocido')
            inst = r.get('institution', '')
            score = r.get('score', 0.0)
            topics = ", ".join(r.get('topics', [])[:2])
            
            logger.info(f"  #{i} {name} (Score: {score:.3f})")
            if inst:
                logger.info(f"     🏛 Institución: {inst}")
            if topics:
                logger.info(f"     🔬 Topics: {topics}")
            logger.info(f"     💡 RAG Reasoning: {r.get('explanation', '')}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Evaluación completada.")

if __name__ == "__main__":
    run_evaluation()
