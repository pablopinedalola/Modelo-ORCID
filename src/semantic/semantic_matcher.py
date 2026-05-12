"""
semantic_matcher.py -- Comparador semantico de perfiles academicos.

Evalua la coherencia conceptual entre un investigador SNII y sus
candidatos ORCID/OpenAlex usando embeddings (sentence-transformers).

No depende solo del nombre: compara areas, disciplinas, keywords,
titulos de publicaciones y conceptos de OpenAlex.

El semantic_score producido se integra al EvidenceVector como la
7ma dimension del modelo de refinamiento iterativo.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.models.schemas import (
    Candidate,
    EvidenceVector,
    NormalizedRecord,
)
from src.semantic.embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """Comparador semantico de perfiles academicos.

    Genera un semantic_score basado en cosine similarity entre
    embeddings del investigador SNII y cada candidato.

    Compara:
        1. Perfil investigador vs. perfil candidato (embedding global)
        2. Disciplina SNII vs. conceptos del candidato
        3. Disciplina SNII vs. titulos de publicaciones del candidato

    El score final es el maximo de todas las comparaciones.

    Attributes:
        engine: Motor de embeddings.

    Examples:
        >>> matcher = SemanticMatcher()
        >>> score = matcher.compute_semantic_score(record, candidate)
        >>> print(f"Semantic: {score:.3f}")
    """

    def __init__(self, engine: Optional[EmbeddingEngine] = None) -> None:
        self.engine = engine or EmbeddingEngine()
        self._researcher_cache: dict[str, object] = {}

    def compute_semantic_score(
        self,
        record: NormalizedRecord,
        candidate: Candidate,
        paper_titles: Optional[list[str]] = None,
    ) -> float:
        """Calcula semantic_score entre investigador y candidato.

        Combina multiples senales de similaridad semantica:
        1. Perfil global (disciplina+area+inst vs. concepts+aff)
        2. Disciplina vs. conceptos individuales
        3. Disciplina vs. titulos de publicaciones

        Args:
            record: Investigador SNII normalizado.
            candidate: Candidato a evaluar.
            paper_titles: Titulos de publicaciones del candidato (opcional).

        Returns:
            Score semantico en [0.0, 1.0].
        """
        scores = []

        # 1. Perfil global vs. perfil global
        profile_sim = self._profile_similarity(record, candidate)
        if profile_sim > 0:
            scores.append(profile_sim)

        # 2. Disciplina vs. conceptos
        concept_sim = self._discipline_vs_concepts(record, candidate)
        if concept_sim > 0:
            scores.append(concept_sim)

        # 3. Disciplina vs. titulos de publicaciones
        if paper_titles:
            pub_sim = self._discipline_vs_publications(record, paper_titles)
            if pub_sim > 0:
                scores.append(pub_sim)

        if not scores:
            return 0.0

        # Score final: promedio ponderado del top
        # (maximo tiene peso doble para recompensar match fuerte)
        best = max(scores)
        avg = sum(scores) / len(scores)
        final = 0.6 * best + 0.4 * avg

        return min(final, 1.0)

    def _profile_similarity(
        self,
        record: NormalizedRecord,
        candidate: Candidate,
    ) -> float:
        """Similitud entre perfil SNII y perfil candidato.

        Compara embedding del investigador (disciplina + area + inst)
        vs. embedding del candidato (concepts + affiliations).

        Args:
            record: Investigador SNII.
            candidate: Candidato.

        Returns:
            Cosine similarity en [0.0, 1.0].
        """
        researcher_emb = self._get_researcher_embedding(record)
        candidate_emb = self.engine.embed_candidate(candidate)

        sim = self.engine.cosine_similarity(researcher_emb, candidate_emb)
        return max(0.0, sim)  # Clamp negatives

    def _discipline_vs_concepts(
        self,
        record: NormalizedRecord,
        candidate: Candidate,
    ) -> float:
        """Similitud entre disciplina SNII y conceptos del candidato.

        Compara el embedding de la disciplina del investigador contra
        cada concepto/topic del candidato y retorna el maximo.

        Args:
            record: Investigador SNII.
            candidate: Candidato.

        Returns:
            Maximo cosine similarity.
        """
        disc = record.original.disciplina
        if not disc or not candidate.concepts:
            return 0.0

        # Embedding de la disciplina SNII
        disc_text = f"{disc} {record.original.area_label}"
        disc_emb = self.engine.embed_text(disc_text)

        if disc_emb is None:
            return 0.0

        best_sim = 0.0
        for concept in candidate.concepts[:10]:
            concept_emb = self.engine.embed_text(concept)
            sim = self.engine.cosine_similarity(disc_emb, concept_emb)
            best_sim = max(best_sim, sim)

        return max(0.0, best_sim)

    def _discipline_vs_publications(
        self,
        record: NormalizedRecord,
        paper_titles: list[str],
    ) -> float:
        """Similitud entre disciplina SNII y titulos de publicaciones.

        Args:
            record: Investigador SNII.
            paper_titles: Titulos de papers del candidato.

        Returns:
            Promedio de top-3 similarities.
        """
        disc = record.original.disciplina
        if not disc or not paper_titles:
            return 0.0

        disc_text = f"{disc} {record.original.area_label}"
        disc_emb = self.engine.embed_text(disc_text)

        if disc_emb is None:
            return 0.0

        sims = []
        for title in paper_titles[:10]:
            if not title:
                continue
            title_emb = self.engine.embed_text(title)
            sim = self.engine.cosine_similarity(disc_emb, title_emb)
            sims.append(max(0.0, sim))

        if not sims:
            return 0.0

        # Promedio de top-3
        sims.sort(reverse=True)
        top = sims[:3]
        return sum(top) / len(top)

    def _get_researcher_embedding(self, record: NormalizedRecord):
        """Obtiene embedding del investigador (con cache)."""
        cache_key = record.id
        if cache_key not in self._researcher_cache:
            self._researcher_cache[cache_key] = self.engine.embed_researcher(record)
        return self._researcher_cache[cache_key]

    def batch_score(
        self,
        record: NormalizedRecord,
        candidates: list[Candidate],
        paper_titles_map: Optional[dict[str, list[str]]] = None,
    ) -> dict[str, float]:
        """Calcula semantic_score para multiples candidatos.

        Args:
            record: Investigador SNII.
            candidates: Lista de candidatos.
            paper_titles_map: Dict {source_id: [titles]}.

        Returns:
            Dict {source_id: semantic_score}.
        """
        results = {}
        for c in candidates:
            titles = (paper_titles_map or {}).get(c.source_id, [])
            score = self.compute_semantic_score(record, c, titles)
            results[c.source_id] = score

        return results
