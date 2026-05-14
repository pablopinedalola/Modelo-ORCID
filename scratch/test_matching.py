
import sys
import os
import json
import logging

# Configurar logging para ver boosts
logging.basicConfig(level=logging.INFO)

# Añadir el directorio raíz al path
sys.path.append(os.getcwd())

from src.rag.hybrid_retriever import HybridRetriever
from src.rag.query_interpreter import QueryInterpreter

def test_search(query):
    print(f"\n{'-'*60}")
    print(f"BÚSQUEDA: {query}")
    print(f"{'-'*60}")
    
    retriever = HybridRetriever()
    print(f"Retriever listo: {retriever.is_ready()}")
    print(f"Perfiles cargados: {len(retriever.profiles)}")
    print(f"Semantic ready: {retriever._semantic_ready}")
    print(f"Lexical ready: {retriever._lexical_ready}")
    
    results = retriever.search(query, top_k=5)
    
    for i, r in enumerate(results):
        print(f"{i+1}. {r['nombre_completo']} ({r['institucion']})")
        print(f"   Score: {r['score']:.4f} | Rank: {r['rank']}")
        print(f"   Boost: {r.get('boost', 0):.2f} | Reason: {r.get('boost_reason', 'N/A')}")
        print(f"   Expl: {r.get('explanation', 'N/A')}")
        print()

if __name__ == "__main__":
    test_search("Miguel Alcubierre")
    test_search("alcubierre")
    test_search("Humberto Carrillo Calvet")
    test_search("Carrillo Calvet")
