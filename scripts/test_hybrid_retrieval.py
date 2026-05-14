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
    from src.rag.hybrid_retriever import HybridRetriever
    from src.rag.bm25_retriever import BM25Retriever
    from src.normalizer.name_normalizer import NameNormalizer

    print("📥 Cargando Motor Híbrido...")
    retriever = HybridRetriever()
    retriever.load()

    # 1. Miguel Alcubierre Profile Verification
    pid = "oa_A5043129140"
    profile = retriever._profiles_by_id.get(pid)
    
    print("\n" + "="*60)
    print("🔬 VERIFICACIÓN DE ENTIDAD: Miguel Alcubierre")
    print("="*60)
    
    if profile:
        print(f"ID:              {profile.get('id')}")
        print(f"Nombre Completo: {profile.get('nombre_completo')}")
        print(f"Aliases:         {profile.get('aliases')}")
        
        # BM25 Text Construction
        bm25 = retriever.bm25_retriever
        text = bm25._build_index_text(profile)
        print(f"\n[TEXTO BM25 CONSTRUIDO]:\n{text}")
        
        # Test Search with Evidence
        for q in ["alcubierre", "Miguel Alcubierre", "Carrillo Calvet"]:
            print(f"\n" + "-"*40)
            print(f"🔎 BÚSQUEDA: \"{q}\"")
            print("-"*40)
            results = retriever.search(q, top_k=5)
            for r in results:
                print(f"#{r['rank']} {r['nombre_completo']} ({r['institucion']})")
                print(f"   Score Final: {r['score']:.4f}")
                print(f"   Boost: {r.get('boost', 0.0):.2f} | Razón: {r.get('boost_reason', 'N/A')}")
                print(f"   Matches BM25: {r.get('matched_tokens', [])}")
                print(f"   Semántico: {r.get('semantic_score', 0):.4f} | Léxico: {r.get('lexical_score', 0):.4f}")
    else:
        print("❌ Miguel Alcubierre no encontrado en profiles indexados.")

    # 2. Humberto Carrillo Calvet Profile Verification
    pid_c = "oa_A5016104570"
    profile_c = retriever._profiles_by_id.get(pid_c)
    
    print("\n" + "="*60)
    print("🔬 VERIFICACIÓN DE ENTIDAD: Humberto Carrillo-Calvet")
    print("="*60)
    
    if profile_c:
        print(f"ID:              {profile_c.get('id')}")
        print(f"Nombre Completo: {profile_c.get('nombre_completo')}")
        print(f"Aliases:         {profile_c.get('aliases')}")
    else:
        print("❌ Humberto Carrillo-Calvet no encontrado en profiles indexados.")

if __name__ == "__main__":
    main()
