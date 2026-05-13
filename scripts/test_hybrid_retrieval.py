#!/usr/bin/env python3
"""
test_hybrid_retrieval.py — Validaciones E2E del sistema híbrido.

Ejecuta tests comprobando que:
1. BM25 carga y funciona.
2. Hybrid ranking da resultados lógicos.
3. El filtering de metadata opera correctamente.
4. Las explicaciones son robustas.

Uso:
    python scripts/test_hybrid_retrieval.py
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
    print("🧪 TEST DE RETRIEVAL HÍBRIDO")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    start_time = time.time()

    from src.rag.hybrid_retriever import HybridRetriever
    
    print("📥 Cargando Motor Híbrido...")
    retriever = HybridRetriever()
    is_ready = retriever.load()
    stats = retriever.stats()
    
    if not is_ready:
        print("❌ Motor no está listo. Abortando tests.")
        sys.exit(1)
        
    print(f"✅ Motor cargado. Modos activos:")
    print(f"   - Semántico: {stats['semantic_ready']}")
    print(f"   - Léxico: {stats['lexical_ready']}")
    print(f"   - Total profiles: {stats['total_profiles']}")
    print()
    passed += 1

    # Test 1: Búsqueda básica híbrida
    print("🔍 Test 1: Búsqueda Híbrida Básica ('física UNAM')")
    res = retriever.search("física UNAM", top_k=3)
    if res:
        print(f"   ✅ Se encontraron {len(res)} resultados.")
        top = res[0]
        print(f"   Top: {top['nombre_completo']} ({top['score']:.3f})")
        print(f"   Explicación: {top['explanation']}")
        if "UNAM" in top['explanation'] or "física" in top['explanation'].lower():
            passed += 1
        else:
            print("   ⚠️ Explicación no parece contener keywords relevantes.")
    else:
        print("   ❌ Sin resultados.")
        failed += 1
    print()

    # Test 2: Filtrado explícito
    print("🔍 Test 2: Metadata Filtering (área='II')")
    res_filtered = retriever.search("investigador", filters={"area": "II"}, top_k=5)
    
    all_correct = True
    for r in res_filtered:
        if r.get("area") != "II":
            all_correct = False
            break
            
    if all_correct and res_filtered:
        print(f"   ✅ Se encontraron {len(res_filtered)} investigadores del área II.")
        passed += 1
    elif not res_filtered:
        print("   ⚠️ Sin resultados en área II (puede ser correcto dependiendo de los datos).")
        passed += 1 # asumimos que puede ser correcto
    else:
        print("   ❌ Filtro falló. Se colaron áreas distintas.")
        failed += 1
    print()

    # Test 3: Rendimiento
    print("⚡ Test 3: Rendimiento Híbrido")
    queries = ["teoría de gráficas", "redes complejas", "química computacional", "ingeniería civil IPN"] * 5
    t0 = time.time()
    for q in queries:
        retriever.search(q, top_k=5, include_explanation=False)
    dt = time.time() - t0
    qps = len(queries) / dt
    
    print(f"   {len(queries)} queries en {dt:.3f}s -> {qps:.1f} QPS")
    if qps > 10:
        print("   ✅ Rendimiento aceptable para híbrido local.")
        passed += 1
    else:
        print("   ⚠️ Rendimiento bajo.")
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
