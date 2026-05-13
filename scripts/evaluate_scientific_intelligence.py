#!/usr/bin/env python3
"""
evaluate_scientific_intelligence.py — Validar métricas y tendencias de inteligencia científica.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from src.graph.knowledge_graph import AcademicKnowledgeGraph
from src.graph.citation_network import CitationNetwork
from src.analytics.temporal_analysis import TemporalAnalyzer
from src.analytics.community_detection import CommunityDetector
from src.analytics.scientific_metrics import ScientificMetrics
from src.analytics.trend_detection import TrendDetector
from src.recommendation.recommender import RecommenderEngine

def evaluate():
    print("=" * 60)
    print("📊 EVALUACIÓN: SCIENTIFIC INTELLIGENCE LAYER")
    print("=" * 60)
    
    graph_file = DATA_DIR / "graph" / "academic_graph.gpickle"
    works_file = DATA_DIR / "openalex" / "works" / "works_cache.json"

    kg = AcademicKnowledgeGraph.load_pickle(str(graph_file))
    cnet = CitationNetwork(kg)
    cnet.build_from_openalex_cache(works_file)

    print("\n1. Métricas de Impacto (Citation Network)")
    metrics = ScientificMetrics(kg.G)
    infl_papers = cnet.get_influential_papers(top_k=5)
    for p in infl_papers:
        print(f"   - '{p['title'][:60]}...': {p['local_citations']} citas locales, {p['global_citations']} globales")

    print("\n2. Productividad Institucional")
    prod = metrics.get_institution_productivity()
    for inst, count in list(prod.items())[:3]:
        print(f"   - {inst}: {count} papers")

    print("\n3. Tendencias Tópicas (Trend Detection)")
    trends = TrendDetector(kg.G)
    t = trends.detect_trending_topics()
    for topic, data in list(t.items())[:5]:
        print(f"   - {topic}: {data['status']} (Slope: {data['slope']:.2f}, Vol: {data['total_recent']})")

    print("\n4. Detección de Comunidades")
    comm = CommunityDetector(kg.G)
    clusters = comm.detect_collaboration_communities()
    if clusters:
        for cid, authors in list(clusters.items())[:3]:
            print(f"   - Comunidad {cid}: {len(authors)} autores -> {', '.join(authors[:3])}...")
    else:
        print("   - No se detectaron comunidades densas en esta pequeña muestra.")

    print("\n5. Hubs Científicos (Centralidad de Autores)")
    hubs = metrics.get_author_centrality(top_k=3)
    for h in hubs:
        print(f"   - {h['author']}: {h['centrality']:.4f}")

    print("\n=" * 60)
    print("✅ EVALUACIÓN COMPLETADA.")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
