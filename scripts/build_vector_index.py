#!/usr/bin/env python3
"""
build_vector_index.py — Construye el índice vectorial FAISS.

Pipeline:
    1. Cargar perfiles procesados (snii_profiles.json)
    2. Generar embeddings con sentence-transformers
    3. Construir índice FAISS (IndexFlatIP + L2 norm = cosine sim)
    4. Persistir todo en data/vector_store/
    5. Validar integridad del índice

Uso:
    python scripts/build_vector_index.py
    python scripts/build_vector_index.py --profiles data/processed/snii_profiles.json
    python scripts/build_vector_index.py --validate-only

Salida en data/vector_store/:
    snii_embeddings.npy       — Embeddings numpy
    snii_metadata.json        — Metadata de perfiles
    snii_cache.pkl            — Cache de embeddings
    snii_faiss.index          — Índice FAISS
    snii_faiss_meta.json      — Metadata del índice
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ajustar PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, EMBEDDING_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_vector_index")


def build_index(
    profiles_path: Path,
    validate_only: bool = False,
) -> dict:
    """Pipeline completo: perfiles → embeddings → FAISS → validación.

    Args:
        profiles_path: Ruta al archivo snii_profiles.json.
        validate_only: Si True, solo valida el índice existente.

    Returns:
        Dict con estadísticas del proceso.
    """
    from src.rag.embedding_pipeline import EmbeddingPipeline
    from src.rag.faiss_store import FAISSStore

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("🔧 CONSTRUCCIÓN DE ÍNDICE VECTORIAL FAISS")
    logger.info("=" * 60)
    logger.info(f"   Modelo:   {EMBEDDING_MODEL}")
    logger.info(f"   Perfiles: {profiles_path}")
    logger.info("")

    store = FAISSStore()

    # ── Modo validación ─────────────────────────────────────────────
    if validate_only:
        return _validate_index(store)

    # ── Paso 1: Cargar perfiles ─────────────────────────────────────
    logger.info("📥 PASO 1/5: Carga de perfiles procesados...")
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    logger.info(f"   ✅ {len(profiles)} perfiles cargados")
    logger.info("")

    # ── Paso 2: Generar embeddings ──────────────────────────────────
    logger.info("🧠 PASO 2/5: Generación de embeddings...")
    pipeline = EmbeddingPipeline()

    # Intentar cargar cache
    pipeline.load_cache()

    result = pipeline.generate_embeddings(
        profiles,
        fields=["searchable_text", "disciplina"],
    )
    logger.info(f"   Embeddings: {result['total']} x {result['dimension']}")
    logger.info("")

    # ── Paso 3: Persistir embeddings ────────────────────────────────
    logger.info("💾 PASO 3/5: Persistencia de embeddings...")
    emb_paths = pipeline.save(result)
    logger.info("")

    # ── Paso 4: Construir índice FAISS ──────────────────────────────
    logger.info("📊 PASO 4/5: Construcción de índice FAISS...")
    store.build_index(
        embeddings=result["embeddings"],
        profile_ids=result["profile_ids"],
        metadata=result["metadata"],
    )
    faiss_paths = store.save()
    logger.info("")

    # ── Paso 5: Validación ──────────────────────────────────────────
    logger.info("✅ PASO 5/5: Validación de integridad...")
    validation = _validate_index(store)

    # ── Estadísticas finales ────────────────────────────────────────
    elapsed = time.time() - start_time

    stats = {
        "total_profiles": len(profiles),
        "total_embeddings": result["total"],
        "dimension": result["dimension"],
        "model": result["model"],
        "faiss_vectors": store.index.ntotal,
        "build_time_seconds": round(elapsed, 2),
        "validation": validation,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RESUMEN DE CONSTRUCCIÓN")
    logger.info("=" * 60)
    logger.info(f"   Perfiles procesados:  {stats['total_profiles']}")
    logger.info(f"   Embeddings generados: {stats['total_embeddings']}")
    logger.info(f"   Dimensión:            {stats['dimension']}")
    logger.info(f"   Vectores en FAISS:    {stats['faiss_vectors']}")
    logger.info(f"   Modelo:               {stats['model']}")
    logger.info(f"   Tiempo:               {stats['build_time_seconds']}s")
    logger.info(f"   Validación:           {'✅ PASSED' if validation.get('all_passed') else '❌ FAILED'}")
    logger.info("")

    # Quick search test
    _run_search_test(pipeline, store)

    logger.info("=" * 60)
    logger.info(f"✅ ÍNDICE CONSTRUIDO EXITOSAMENTE en {elapsed:.2f}s")
    logger.info("=" * 60)

    return stats


def _validate_index(store) -> dict:
    """Valida integridad del índice FAISS.

    Args:
        store: FAISSStore con índice cargado o construido.

    Returns:
        Dict con resultados de validación.
    """
    # Si no hay índice en memoria, intentar cargar
    if store.index is None:
        if not store.load():
            logger.error("   ❌ No se pudo cargar el índice FAISS")
            return {"all_passed": False, "error": "Index not found"}

    checks = store.integrity_check()

    for check_name, passed in checks.items():
        if check_name == "all_passed":
            continue
        status = "✅" if passed else "❌"
        logger.info(f"   {status} {check_name}: {passed}")

    if checks["all_passed"]:
        logger.info(f"   ✅ Todas las verificaciones pasaron")
    else:
        logger.warning(f"   ❌ Algunas verificaciones fallaron")

    return checks


def _run_search_test(pipeline, store):
    """Ejecuta búsquedas de prueba para verificar el índice.

    Args:
        pipeline: EmbeddingPipeline para generar query embeddings.
        store: FAISSStore con índice construido.
    """
    test_queries = [
        "física de partículas",
        "matemáticas aplicadas UNAM",
        "ingeniería mecatrónica",
        "bioquímica universidad guadalajara",
        "óptica cuántica",
    ]

    logger.info("")
    logger.info("🔍 BÚSQUEDAS DE PRUEBA:")
    logger.info("-" * 60)

    for query in test_queries:
        query_vector = pipeline.encode_query(query)
        results = store.search(query_vector, top_k=3, min_score=0.0)

        logger.info(f"   Q: \"{query}\"")
        if results:
            for r in results:
                logger.info(
                    f"      #{r['rank']} {r['nombre_completo'][:35]:35s} "
                    f"score={r['score']:.4f}  "
                    f"[{r['disciplina'][:25]}]"
                )
        else:
            logger.info("      (sin resultados)")
        logger.info("")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Construye el índice vectorial FAISS para perfiles SNII",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=PROCESSED_DATA_DIR / "snii_profiles.json",
        help="Ruta al archivo de perfiles procesados",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Solo validar el índice existente sin reconstruir",
    )

    args = parser.parse_args()

    if not args.profiles.exists() and not args.validate_only:
        logger.error(f"❌ Archivo de perfiles no encontrado: {args.profiles}")
        logger.info("   Ejecuta primero: python scripts/ingest_snii.py <archivo_snii>")
        sys.exit(1)

    try:
        build_index(
            profiles_path=args.profiles,
            validate_only=args.validate_only,
        )
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
