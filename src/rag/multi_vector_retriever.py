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

from src.rag.query_interpreter import QueryInterpreter, INSTITUTIONS
from src.rag.embedding_pipeline import EmbeddingPipeline
from src.rag.faiss_store import FAISSStore
from src.rag.bm25_retriever import BM25Retriever
from api.openalex_data import get_all_works, get_authors, search_topics

logger = logging.getLogger(__name__)

class MultiVectorRetriever:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.interpreter = QueryInterpreter()
        self.faiss_store = FAISSStore(store_dir=self.store_dir)
        self.embedding_pipeline = EmbeddingPipeline(store_dir=self.store_dir)
        self.bm25_retriever = BM25Retriever(store_dir=self.store_dir)
        
        self.authors = []
        self._authors_by_id = {}
        self.works = []
        self.snii_profiles = {}
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
            
            # Cargar SNII profiles from metadata in FAISS
            if self.faiss_store.metadata:
                self.snii_profiles = {self.faiss_store.profile_ids[i]: meta for i, meta in enumerate(self.faiss_store.metadata)}
            
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

        # 1. Query Interpretation
        interpretation = self.interpreter.interpret(query)
        rewritten_query = interpretation["rewritten_query"]
        intent = interpretation["intent"]
        target_institutions = interpretation["institutions"]
        expanded_concepts = interpretation["expanded_concepts"]
        
        print("\n" + "="*50)
        print("🔍 RUNTIME LOG: QUERY INTERPRETER")
        print("="*50)
        print(f"[LOG] Original Query: '{query}'")
        print(f"[LOG] Normalized Query: '{interpretation.get('normalized_query', query.lower())}'")
        print(f"[LOG] Institution Detected: {target_institutions}")
        print(f"[LOG] Semantic Expansions: {expanded_concepts}")
        print(f"[LOG] Expanded Query (Dense): '{rewritten_query}'")
        print("="*50 + "\n")
        
        logger.info(f"Natural Language Query: '{query}'")
        logger.info(f" -> Intent: {intent}")
        logger.info(f" -> Institutions: {target_institutions}")
        logger.info(f" -> Rewritten for Dense: '{rewritten_query}'")

        # 2. Semantic Search (Dense)
        query_vector = self.embedding_pipeline.encode_query(rewritten_query)
        semantic_results = {}
        if self.faiss_store.index is not None:
            sem_res = self.faiss_store.search(query_vector, top_k=top_k * 4)
            semantic_results = {r["profile_id"]: r["score"] for r in sem_res}

        # 3. Lexical Search (Sparse BM25)
        lexical_results = {}
        if self.bm25_retriever.bm25 is not None:
            lex_res = self.bm25_retriever.search(rewritten_query, top_k=top_k * 4)
            lexical_results = {r["profile_id"]: r["score"] for r in lex_res}

        # 4. Synthesize Candidates
        fused_scores = {}
        all_candidate_ids = set(semantic_results.keys()) | set(lexical_results.keys())
        
        print(f"\n[LOG] Top-k Retrieval RAW: {len(all_candidate_ids)} candidatos encontrados (Semantic + Lexical)")

        # Add matches based on OpenAlex explicit topics matching the expanded concepts
        for concept in expanded_concepts:
            topics = search_topics(concept)
            for a in self.authors:
                a_topics = [at.get("display_name", "").lower() for at in a.get("topics", [])]
                if any(concept in at for at in a_topics):
                    all_candidate_ids.add(a.get("_slug"))

        results = []
        for cid in all_candidate_ids:
            if not cid: continue
            
            s_score = semantic_results.get(cid, 0.0)
            l_score = lexical_results.get(cid, 0.0)
            l_score_norm = min(l_score / 15.0, 1.0)
            
            # Determine if it's an author or a work
            is_work = cid.startswith("work_")
            
            if is_work:
                # Basic work mock scoring
                final_score = (s_score * 0.5) + (l_score_norm * 0.5)
                if final_score < 0.1: continue
                
                reasons = []
                match_types = []
                if s_score > 0.4:
                    reasons.append(f"Paper: Match Semántico ({s_score:.2f})")
                    match_types.append("semantic")
                if l_score > 0:
                    reasons.append("Paper: Match Léxico BM25")
                    match_types.append("lexical")
                    
                # Reranking boost based on intent
                if intent == "paper":
                    final_score += 0.2
                    reasons.append("Boost: Intención de búsqueda de papers")

                results.append({
                    "profile_id": cid,
                    "slug": cid,
                    "display_name": f"Paper ID {cid.split('_')[-1]}", # Simplified
                    "institution": "",
                    "score": final_score,
                    "match_types": match_types,
                    "explanation": " | ".join(reasons),
                    "topics": []
                })
                continue
                
            # It's an author (SNII Profile)
            author_data = self.snii_profiles.get(cid, {})
            if not author_data: continue
            
            # 5. Institution Filtering
            inst_name = author_data.get("institucion", "").lower()
            import unicodedata
            inst_name_norm = "".join(c for c in unicodedata.normalize("NFKD", inst_name) if not unicodedata.combining(c))
            inst_pass = True
            if target_institutions:
                inst_pass = False
                for target_inst in target_institutions:
                    aliases = INSTITUTIONS.get(target_inst, [target_inst])
                    if any(alias in inst_name_norm for alias in aliases) or (target_inst == "unam" and "autonoma de mexico" in inst_name_norm):
                        inst_pass = True
                        break
                        
            if not inst_pass:
                continue # Skip if it doesn't match the requested institution

            # Feature: Topic Overlap
            topic_overlap_score = 0.0
            matched_topics = []
            a_topics = [author_data.get("disciplina", "").lower()]
            for ext in expanded_concepts:
                for at in a_topics:
                    if ext in at or at in ext:
                        topic_overlap_score += 0.2
                        matched_topics.append(at)

            # Final Score Fusion
            final_score = (s_score * 0.4) + (l_score_norm * 0.3) + min(topic_overlap_score, 0.3)
            
            if final_score < 0.1:
                continue

            # Build reasoning/explainability
            reasons = []
            match_types = []
            
            if s_score > 0.4:
                reasons.append(f"Similitud semántica densa ({s_score:.2f})")
                match_types.append("semantic")
            if l_score > 0:
                reasons.append(f"Coincidencia de palabras clave BM25 ({l_score_norm:.2f})")
                match_types.append("lexical")
            if matched_topics:
                reasons.append(f"Match en disciplina: {', '.join(set(matched_topics))}")
                match_types.append("topic")
            if target_institutions and inst_pass:
                reasons.append(f"Filtro institucional aplicado: {target_institutions[0].upper()}")
                match_types.append("institution")
                final_score += 0.15 # Reranking boost
                
            # Reranking boost based on intent
            if intent == "author":
                final_score += 0.1
                reasons.append("Boost: Intención de búsqueda de autores")
                
            results.append({
                "profile_id": cid,
                "slug": cid,
                "display_name": author_data.get("nombre_completo", cid),
                "institution": author_data.get("institucion", ""),
                "score": final_score,
                "match_types": match_types,
                "explanation": " | ".join(reasons),
                "topics": a_topics[:3]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"[LOG] Results AFTER institution filtering: {len([r for r in results if not r['profile_id'].startswith('work_')])} autores válidos")
        
        return results[:top_k]

