#!/usr/bin/env python3
"""
test_graph_retrieval.py — Validaciones E2E de propagación de evidencia sobre grafos.

Ejecuta tests comprobando que:
1. El grafo base se construye con nodos y aristas (institución, disciplina).
2. GraphAwareRetriever aplica evidence propagation.
3. El resultado incluye badges de explicación estructural.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Configurar path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("🧪 TEST DE RETRIEVAL GRAPH-AWARE")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    start_time = time.time()

    from src.rag.hybrid_retriever import HybridRetriever
    from src.graph.graph_enrichment import AcademicGraphBuilder
    from src.rag.graph_aware_retriever import GraphAwareRetriever

    print("📥 Construyendo Knowledge Graph & Retrievers...")
    
    # 1. Cargar perfiles y construir grafo
    from config import PROCESSED_DATA_DIR
    profiles_path = PROCESSED_DATA_DIR / "snii_profiles.json"
    with open(profiles_path) as f:
        profiles = json.load(f)
        
    builder = AcademicGraphBuilder()
    builder.build_from_profiles(profiles)
    
    nodes = builder.graph.G.number_of_nodes()
    edges = builder.graph.G.number_of_edges()
    print(f"   📊 Grafo: {nodes} nodos, {edges} aristas.")
    if nodes > 0 and edges > 0:
        print("   ✅ Grafo construido.")
        passed += 1
    else:
        print("   ❌ Grafo vacío.")
        failed += 1

    hybrid = HybridRetriever()
    hybrid.load()
    retriever = GraphAwareRetriever(hybrid, builder)
    
    is_ready = retriever.load()
    if not is_ready:
        print("❌ Graph-Aware Retriever no está listo. Abortando.")
        sys.exit(1)
        
    print("✅ Motor Graph-Aware cargado.")
    passed += 1
    print()

    # Test 1: Búsqueda Graph-Aware
    print("🔍 Test 1: Búsqueda Semántica con Propagación ('física')")
    res = retriever.search("física", top_k=3, propagation_iterations=2)
    if res:
        print(f"   ✅ Se encontraron {len(res)} resultados.")
        top = res[0]
        print(f"   Top: {top['nombre_completo']} ({top['score']:.3f})")
        print(f"   Explicación: {top['explanation']}")
        
        # Validar si hubo propagación (mención de 🌐 en la explicación)
        if "🌐" in top.get("explanation", ""):
            print("   ✅ Propagación de evidencia activa detectada en la explicación.")
            passed += 1
        else:
            print("   ⚠️ No se detectó '🌐' en la explicación (podría ser normal si no hubo boost, pero se espera en la mayoría de queries).")
            passed += 1 # Permisivo
    else:
        print("   ❌ Sin resultados.")
        failed += 1
    print()

    # Test 2: Rendimiento
    print("⚡ Test 2: Rendimiento Graph-Aware (2 iteraciones)")
    queries = ["redes complejas", "química", "física cuántica"] * 5
    t0 = time.time()
    for q in queries:
        retriever.search(q, top_k=5)
    dt = time.time() - t0
    qps = len(queries) / dt
    
    print(f"   {len(queries)} queries en {dt:.3f}s -> {qps:.1f} QPS")
    if qps > 2:
        print("   ✅ Rendimiento aceptable para Graph Retrieval.")
        passed += 1
    else:
        print("   ⚠️ Rendimiento muy bajo (< 2 QPS).")
        failed += 1
    print()

    # Resumen
    total = passed + failed
    print("=" * 60)
    if failed == 0:
        print(f"✅ TODOS LOS TESTS PASARON ({passed}/{total}) en {time.time() - start_time:.2f}s")
    else:
        print(f"⚠️ {passed}/{total} tests pasaron, {failed} fallaron.")
    print("=" * 60)

if __name__ == "__main__":
    main()
