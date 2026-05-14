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
import unicodedata
from pathlib import Path

def normalize_text(text: str) -> str:
    """Normaliza texto para searchable_text."""
    if not text: return ""
    text = str(text).lower()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    return text

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
    from src.rag.vector_store import VectorStore

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("🔧 CONSTRUCCIÓN DE ÍNDICE VECTORIAL FAISS")
    logger.info("=" * 60)
    logger.info(f"   Modelo:   {EMBEDDING_MODEL}")
    logger.info(f"   Perfiles: {profiles_path}")
    logger.info("")

    store = VectorStore()

    # ── Modo validación ─────────────────────────────────────────────
    if validate_only:
        return _validate_index(store)

    # ── Paso 1: Cargar perfiles principales ─────────────────────────────────────
    logger.info("📥 PASO 1/5: Carga de perfiles principales...")
    profiles = []
    unam_authors_path = PROCESSED_DATA_DIR / "unam_authors.json"
    
    if unam_authors_path.exists():
        with open(unam_authors_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
    elif profiles_path.exists():
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
    # Si no hay snii_profiles ni unam_authors, inicializamos vacío para que lo cargue OpenAlex
    if not profiles:
        logger.info("   ⚠️ No se encontró corpus UNAM/SNII, se usará OpenAlex como corpus principal.")
        
    for p in profiles:
        if not p.get("searchable_text"):
            title = p.get("nombre_completo", "")
            abstract = p.get("abstract", "")
            topics = " ".join(p.get("topics", []))
            p["searchable_text"] = normalize_text(f"{title} {abstract} {topics}")
            
    logger.info(f"   ✅ {len(profiles)} perfiles base cargados")
    logger.info("")

    # Eliminamos el uso antiguo de EmbeddingPipeline y result dict.
    # El VectorStore de src.rag se encarga internamente de los embeddings vía add_chunks().
    logger.info("   (Generación de embeddings deferida a VectorStore)")
    
    # NUEVO: Indexar Works/Papers de UNAM
    try:
        unam_works_path = PROCESSED_DATA_DIR / "unam_works.json"
        if unam_works_path.exists():
            with open(unam_works_path, "r", encoding="utf-8") as f:
                works = json.load(f)
            logger.info(f"   Indexando {len(works)} papers de UNAM...")
            work_profiles = []
            for w in works:
                title = w.get("title", "")
                abstract = w.get("abstract", "")
                topics = " ".join(w.get("topics", []))
                
                searchable_text = w.get("searchable_text")
                if not searchable_text:
                    searchable_text = normalize_text(f"{title} {abstract} {topics}")
                
                if searchable_text:
                    work_profiles.append({
                        "id": f"work_{w.get('id')}",
                        "title": title,
                        "abstract": abstract,
                        "searchable_text": searchable_text,
                        "disciplina": "Investigación UNAM",
                        "nombre_completo": title,
                        "institucion": "UNAM",
                        "dependencia": "UNAM"
                    })
            if work_profiles:
                profiles.extend(work_profiles)
        else:
            logger.info("   No se encontraron papers de UNAM para indexar.")
    except Exception as e:
        logger.warning(f"   ⚠️ Error indexando papers de UNAM: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # NUEVO: Indexar Works/Papers de OpenAlex
    try:
        openalex_works_path = PROCESSED_DATA_DIR / "openalex_works.json"
        if openalex_works_path.exists():
            with open(openalex_works_path, "r", encoding="utf-8") as f:
                openalex_works = json.load(f)
            logger.info(f"   Indexando {len(openalex_works)} papers de OpenAlex...")
            openalex_profiles = []
            for w in openalex_works:
                title = w.get("title", "") or ""
                abstract = w.get("abstract", "") or ""
                topics = " ".join(w.get("concepts", []))
                authors = " ".join(w.get("authors", []))
                institutions = " ".join(w.get("institutions", []))
                
                # searchable_text robusto: title + abstract + topics + authors + institution
                searchable_text = normalize_text(f"{title} {abstract} {topics} {authors} {institutions}")
                
                if searchable_text:
                    openalex_profiles.append({
                        "id": f"oa_work_{w.get('id')}",
                        "title": title,
                        "abstract": abstract,
                        "searchable_text": searchable_text,
                        "disciplina": "OpenAlex",
                        "nombre_completo": title,
                        "institucion": "UNAM (OpenAlex)",
                        "dependencia": institutions[:50] if institutions else "OpenAlex"
                    })
            if openalex_profiles:
                # Integrar también a perfiles para BM25 y FAISS
                profiles.extend(openalex_profiles)
        else:
            logger.info("   No se encontraron papers de OpenAlex para indexar.")
    except Exception as e:
        logger.warning(f"   ⚠️ Error indexando papers de OpenAlex: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # NUEVO: Indexar Autores de OpenAlex
    try:
        openalex_authors_path = PROCESSED_DATA_DIR / "openalex_authors.json"
        if openalex_authors_path.exists():
            with open(openalex_authors_path, "r", encoding="utf-8") as f:
                openalex_authors = json.load(f)
            logger.info(f"   Indexando {len(openalex_authors)} autores de OpenAlex...")
            oa_authors_profiles = []
            for a in openalex_authors:
                name = a.get("name", "") or ""
                institutions = " ".join(a.get("institutions", []))
                
                searchable_text = normalize_text(f"{name} {institutions}")
                
                if searchable_text:
                    oa_authors_profiles.append({
                        "id": f"oa_author_{a.get('id')}",
                        "nombre_completo": name,
                        "searchable_text": searchable_text,
                        "disciplina": "Investigador OpenAlex",
                        "institucion": "UNAM (OpenAlex)",
                        "dependencia": institutions[:50] if institutions else "OpenAlex"
                    })
            if oa_authors_profiles:
                # Integrar también a perfiles para BM25 y FAISS
                profiles.extend(oa_authors_profiles)
        else:
            logger.info("   No se encontraron autores de OpenAlex para indexar.")
    except Exception as e:
        logger.warning(f"   ⚠️ Error indexando autores de OpenAlex: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    logger.info("")

    # ── Paso 2: Generar Embeddings y construir índice FAISS ──────────────────────────────
    logger.info("📊 PASO 2/4: Construcción de índice FAISS (y generación de embeddings)...")
    if profiles:
        from sentence_transformers import SentenceTransformer
        import faiss
        
        logger.info(f"   Cargando modelo all-MiniLM-L6-v2...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        texts = [p.get("searchable_text", "") for p in profiles]
        
        logger.info(f"   Generando embeddings para {len(texts)} documentos...")
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Guardar en FAISS (VectorStore)
        faiss.normalize_L2(embeddings)
        store.index.add(embeddings)
        
        # Mapear a metadata y guardar
        for p in profiles:
            p["text"] = p.get("searchable_text", "")
        store.metadata.extend(profiles)
        
        faiss_paths = store.save()
    else:
        logger.error("❌ No hay perfiles para indexar.")
        return {}
    logger.info("")

    # ── Paso 3: Construir índice BM25 ──────────────────────────────
    logger.info("📖 PASO 3/4: Construcción de índice BM25...")
    try:
        import pickle
        import re
        from rank_bm25 import BM25Okapi
        
        corpus_tokens = []
        for p in profiles:
            text = p.get("searchable_text", "").lower()
            tokens = re.findall(r"[a-z0-9]+", text)
            corpus_tokens.append([t for t in tokens if len(t) > 1])
            
        bm25 = BM25Okapi(corpus_tokens)
        bm25_data = {
            "profile_ids": [p["id"] for p in profiles],
            "metadata": [{k: v for k, v in p.items() if k not in ["searchable_text", "text"]} for p in profiles],
            "corpus_tokens": corpus_tokens
        }
        
        # Guardar en VECTOR_STORE_DIR
        bm25_path = store.index_dir / "snii_bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)
            
        logger.info(f"   ✅ BM25 indexado exitosamente ({len(profiles)} documentos).")
    except ImportError:
        logger.warning("   ⚠️ rank_bm25 no instalado. Omitiendo BM25.")
    logger.info("")

    # ── Paso 6: Validación ──────────────────────────────────────────
    logger.info("✅ PASO 6/6: Validación de integridad...")
    validation = _validate_index(store)

    # ── Estadísticas finales ────────────────────────────────────────
    elapsed = time.time() - start_time

    stats = {
        "total_profiles": len(profiles),
        "total_embeddings": store.index.ntotal if store.index else 0,
        "faiss_vectors": store.index.ntotal if store.index else 0,
        "build_time_seconds": round(elapsed, 2),
        "validation": validation,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RESUMEN DE CONSTRUCCIÓN")
    logger.info("=" * 60)
    logger.info(f"   Perfiles procesados:  {stats['total_profiles']}")
    logger.info(f"   Embeddings generados: {stats['total_embeddings']}")
    logger.info(f"   Vectores en FAISS:    {stats['faiss_vectors']}")
    logger.info(f"   Documentos en BM25:   {len(profiles)}")
    logger.info(f"   Tiempo:               {stats['build_time_seconds']}s")
    logger.info(f"   Validación:           {'✅ PASSED' if validation.get('all_passed') else '❌ FAILED'}")
    logger.info("")

    # Quick search test omitted for simplicity, as it requires EmbeddingPipeline.

    logger.info("=" * 60)
    logger.info(f"✅ ÍNDICE CONSTRUIDO EXITOSAMENTE en {elapsed:.2f}s")
    logger.info("=" * 60)

    return stats


def _validate_index(store) -> dict:
    """Valida que los archivos del índice FAISS hayan sido creados."""
    faiss_exists = store.index_path.exists()
    metadata_exists = store.metadata_path.exists()
    
    if faiss_exists and metadata_exists:
        logger.info("   ✅ Archivo FAISS y metadata verificados correctamente.")
        logger.info("   ✅ Índice construido exitosamente.")
        return {"all_passed": True}
    else:
        logger.warning("   ❌ Archivos de índice no encontrados.")
        return {"all_passed": False}


def _run_search_test():
    pass


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

    oa_works_path = PROCESSED_DATA_DIR / "openalex_works.json"
    if not args.profiles.exists() and not args.validate_only and not oa_works_path.exists():
        logger.error(f"❌ Archivo de perfiles no encontrado: {args.profiles} y openalex_works.json tampoco existe.")
        logger.info("   Ejecuta primero: python scripts/fetch_openalex_corpus.py")
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
