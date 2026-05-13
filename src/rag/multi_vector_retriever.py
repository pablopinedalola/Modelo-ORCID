"""
multi_vector_retriever.py — Retriever avanzado Multi-Vector y Multi-Señal.

Implementa un Academic RAG REAL que fusiona:
- Dense semantic score (MPNet multilingüe)
- BM25 lexical score
- Topic overlap (OpenAlex topics/concepts)
- Query Understanding (expansión de keywords, traducción)

Retorna explicabilidad real (reasoning) para cada match.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.rag.query_understanding import QueryAnalyzer
from src.rag.embedding_pipeline import EmbeddingPipeline
from src.rag.faiss_store import FAISSStore
from src.rag.bm25_retriever import BM25Retriever
from api.openalex_data import get_all_works, get_authors, search_topics

logger = logging.getLogger(__name__)

class MultiVectorRetriever:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.analyzer = QueryAnalyzer()
        self.faiss_store = FAISSStore(store_dir=self.store_dir)
        self.embedding_pipeline = EmbeddingPipeline(store_dir=self.store_dir)
        self.bm25_retriever = BM25Retriever(store_dir=self.store_dir)
        
        self.authors = []
        self._authors_by_id = {}
        self.works = []
        self.ready = False

    def load(self) -> bool:
        """Carga índices vectoriales y datos de OpenAlex en memoria."""
        try:
            # Cargar índices
            faiss_ok = self.faiss_store.load()
            bm25_ok = self.bm25_retriever.load()
            
            # Cargar OpenAlex Data
            self.authors = get_authors()
            self._authors_by_id = {a.get("_slug", ""): a for a in self.authors}
            self.works = get_all_works()
            
            if faiss_ok or bm25_ok:
                self.ready = True
                logger.info("MultiVectorRetriever: Índices y datos cargados exitosamente.")
            else:
                logger.warning("MultiVectorRetriever: No se pudieron cargar los índices.")
            return self.ready
        except Exception as e:
            logger.error(f"Error cargando MultiVectorRetriever: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Realiza la búsqueda multicapa y devuelve matches con explicabilidad."""
        if not self.ready:
            logger.warning("MultiVectorRetriever no está listo. Ejecuta load() primero.")
            return []

        # 1. Query Understanding
        q_analysis = self.analyzer.analyze(query)
        expanded_query = q_analysis["search_string"]
        logger.info(f"Query expansion: '{query}' -> '{expanded_query}'")

        # 2. Semantic Search (Dense)
        query_vector = self.embedding_pipeline.encode_query(expanded_query)
        semantic_results = {}
        if self.faiss_store.index is not None:
            sem_res = self.faiss_store.search(query_vector, top_k=top_k * 3)
            # FAISS results use profile_id. Assuming profile_id corresponds to author slug or works.
            semantic_results = {r["profile_id"]: r["score"] for r in sem_res}

        # 3. Lexical Search (Sparse BM25)
        lexical_results = {}
        if self.bm25_retriever.corpus:
            lex_res = self.bm25_retriever.search(expanded_query, top_k=top_k * 3)
            lexical_results = {r["profile_id"]: r["score"] for r in lex_res}

        # 4. OpenAlex Topic Overlap & Synthesis
        fused_scores = {}
        all_candidate_ids = set(semantic_results.keys()) | set(lexical_results.keys())
        
        # Add matches based purely on OpenAlex explicit topics matching the query
        for t in q_analysis["detected_topics"]:
            topics = search_topics(t)
            # Find authors that have these topics
            for a in self.authors:
                a_topics = [at.get("display_name", "").lower() for at in a.get("topics", [])]
                if any(t.lower() in at for at in a_topics):
                    all_candidate_ids.add(a.get("_slug"))

        results = []
        for cid in all_candidate_ids:
            if not cid: continue
            
            s_score = semantic_results.get(cid, 0.0)
            l_score = lexical_results.get(cid, 0.0)
            
            # Normalize BM25 score roughly
            l_score_norm = min(l_score / 15.0, 1.0)
            
            author_data = self._authors_by_id.get(cid, {})
            
            # Feature: Topic Overlap
            topic_overlap_score = 0.0
            matched_topics = []
            a_topics = [at.get("display_name", "").lower() for at in author_data.get("topics", [])]
            for ext in q_analysis["expanded_terms"]:
                for at in a_topics:
                    if ext in at or at in ext:
                        topic_overlap_score += 0.2
                        matched_topics.append(at)

            # Feature: Concept Overlap
            matched_concepts = []
            a_concepts = [c.get("display_name", "").lower() for c in author_data.get("x_concepts", [])]
            for ext in q_analysis["expanded_terms"]:
                for ac in a_concepts:
                    if ext in ac or ac in ext:
                        topic_overlap_score += 0.1
                        matched_concepts.append(ac)

            # Final Score Fusion
            final_score = (s_score * 0.4) + (l_score_norm * 0.3) + min(topic_overlap_score, 0.3)
            
            if final_score < 0.1:
                continue

            # Build reasoning/explainability
            reasons = []
            match_types = []
            
            if s_score > 0.5:
                reasons.append(f"Similitud semántica densa ({s_score:.2f})")
                match_types.append("semantic")
            if l_score > 0:
                reasons.append("Coincidencia de palabras clave (BM25)")
                match_types.append("lexical")
            if matched_topics:
                reasons.append(f"Match en temas OpenAlex: {', '.join(set(matched_topics))}")
                match_types.append("topic")
            if matched_concepts:
                reasons.append(f"Match en conceptos: {', '.join(set(matched_concepts))}")
                match_types.append("concept")
                
            results.append({
                "profile_id": cid,
                "slug": cid,
                "display_name": author_data.get("display_name", cid),
                "institution": author_data.get("last_known_institutions", [{}])[0].get("display_name", "") if author_data.get("last_known_institutions") else "",
                "score": final_score,
                "match_types": match_types,
                "explanation": " | ".join(reasons),
                "topics": a_topics[:3]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

