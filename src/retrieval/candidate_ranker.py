"""
candidate_ranker.py -- Ranking inicial de candidatos por similitud.

Calcula scores de similitud entre un investigador SNII normalizado
y cada candidato ORCID/OpenAlex usando fuzzy matching (rapidfuzz).

Score formula (Fase 2):
    score = 0.7 * name_similarity + 0.3 * institution_similarity

NO usa embeddings, LLM, ni refinamiento iterativo.
Esos se agregan en fases posteriores.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz
except ImportError:
    logger.warning("rapidfuzz not installed. Using basic difflib fallback.")
    import difflib
    class MockFuzz:
        def ratio(self, s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100
        def partial_ratio(self, s1, s2):
            # Very basic fallback
            if s1 in s2 or s2 in s1: return 100
            return self.ratio(s1, s2)
        def token_sort_ratio(self, s1, s2):
            return self.ratio(" ".join(sorted(s1.split())), " ".join(sorted(s2.split())))
        def token_set_ratio(self, s1, s2):
            return self.ratio(s1, s2) # simple fallback
    fuzz = MockFuzz()

from src.models.schemas import (
    Candidate,
    EvidenceVector,
    NormalizedRecord,
    RetrievalResult,
    CandidateSource,
)
from src.normalizer.name_normalizer import NameNormalizer

# Instancia compartida para normalizar nombres de candidatos
_name_normalizer = NameNormalizer()


class CandidateRanker:
    """Rankea candidatos por similitud con el investigador SNII.

    Calcula scores de nombre e institucion usando rapidfuzz y
    ordena candidatos por score combinado. Tambien filtra candidatos
    por debajo de un umbral minimo.

    Attributes:
        name_weight: Peso del score de nombre (default: 0.7).
        institution_weight: Peso del score de institucion (default: 0.3).
        min_score_threshold: Score minimo para retener candidato.

    Examples:
        >>> ranker = CandidateRanker()
        >>> ranked = ranker.rank(normalized_record, candidates)
        >>> for c in ranked:
        ...     print(f"{c.display_name}: {c.confidence:.2f}")
    """

    def __init__(
        self,
        name_weight: float = 0.7,
        institution_weight: float = 0.3,
        min_score_threshold: float = 0.30,
    ) -> None:
        self.name_weight = name_weight
        self.institution_weight = institution_weight
        self.min_score_threshold = min_score_threshold

    # Phase 2 weights: only name and institution are computed
    PHASE2_WEIGHTS = {
        "name": 0.7,
        "institution": 0.3,
        "area": 0.0,
        "publication": 0.0,
        "coauthor": 0.0,
        "temporal": 0.0,
        "semantic": 0.0,
    }

    def rank(
        self,
        record: NormalizedRecord,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Rankea y filtra candidatos para un investigador.

        Calcula name_score e institution_score para cada candidato,
        actualiza el EvidenceVector, y retorna lista ordenada.

        Uses Phase 2 weights (0.7 name + 0.3 institution) so scores
        aren't diluted by zero-valued dimensions from later phases.

        Args:
            record: Investigador SNII normalizado.
            candidates: Lista de candidatos sin rankear.

        Returns:
            Lista de candidatos ordenados por confidence descendente,
            filtrados por umbral minimo.
        """
        if not candidates:
            return []

        for candidate in candidates:
            name_score = self._compute_name_score(record, candidate)
            inst_score = self._compute_institution_score(record, candidate)

            # Actualizar evidence vector
            candidate.evidence = EvidenceVector(
                name_score=name_score,
                institution_score=inst_score,
            )

        # Usar Phase 2 weights para confidence
        get_conf = lambda c: c.evidence.confidence(self.PHASE2_WEIGHTS)

        # Filtrar por umbral
        filtered = [
            c for c in candidates
            if get_conf(c) >= self.min_score_threshold
        ]

        # Ordenar por confidence descendente
        filtered.sort(key=get_conf, reverse=True)

        logger.info(
            f"  Ranker: {len(filtered)}/{len(candidates)} candidatos "
            f"sobre umbral {self.min_score_threshold}"
        )

        return filtered

    def _compute_name_score(
        self,
        record: NormalizedRecord,
        candidate: Candidate,
    ) -> float:
        """Calcula similitud de nombre entre SNII y candidato.

        Usa multiples estrategias y toma el maximo:
        1. Fuzzy ratio entre nombres normalizados
        2. Fuzzy partial_ratio para coincidencia parcial
        3. Token sort ratio para orden diferente
        4. Mejor score contra todos los aliases del SNII

        Args:
            record: Registro SNII normalizado.
            candidate: Candidato a evaluar.

        Returns:
            Score en [0.0, 1.0].
        """
        candidate_name = f"{candidate.given_name} {candidate.family_name}".strip()
        if not candidate_name:
            candidate_name = candidate.display_name

        candidate_normalized = _name_normalizer.normalize(candidate_name)
        snii_normalized = record.normalized_name

        if not candidate_normalized or not snii_normalized:
            return 0.0

        # Estrategia 1: ratio directo
        ratio = fuzz.ratio(snii_normalized, candidate_normalized) / 100.0

        # Estrategia 2: partial ratio (para coincidencias parciales)
        partial = fuzz.partial_ratio(snii_normalized, candidate_normalized) / 100.0

        # Estrategia 3: token sort (ignora orden)
        token_sort = fuzz.token_sort_ratio(snii_normalized, candidate_normalized) / 100.0

        # Estrategia 4: token set (subconjuntos)
        token_set = fuzz.token_set_ratio(snii_normalized, candidate_normalized) / 100.0

        # Estrategia 5: mejor score contra aliases SNII
        best_alias_score = 0.0
        for alias in record.name_aliases[:8]:
            alias_norm = _name_normalizer.normalize(alias)
            if alias_norm:
                s = fuzz.token_sort_ratio(alias_norm, candidate_normalized) / 100.0
                best_alias_score = max(best_alias_score, s)

        # Score final: maximo de todas las estrategias con peso
        scores = [ratio, partial * 0.9, token_sort, token_set * 0.95, best_alias_score]
        return max(scores)

    def _compute_institution_score(
        self,
        record: NormalizedRecord,
        candidate: Candidate,
    ) -> float:
        """Calcula similitud de institucion entre SNII y candidato.

        Compara la institucion normalizada del SNII contra todas
        las afiliaciones del candidato.

        Args:
            record: Registro SNII normalizado.
            candidate: Candidato a evaluar.

        Returns:
            Score en [0.0, 1.0].
        """
        if not candidate.affiliations:
            return 0.0

        snii_institution = record.normalized_institution.lower().strip()
        snii_aliases = [a.lower().strip() for a in record.institution_aliases]

        if not snii_institution:
            return 0.0

        best_score = 0.0

        for affiliation in candidate.affiliations:
            aff_lower = affiliation.lower().strip()

            # Check exacto contra aliases
            if aff_lower in snii_aliases or snii_institution == aff_lower:
                return 1.0

            # Check si la abreviatura aparece en la afiliacion
            for alias in snii_aliases:
                if len(alias) <= 10:  # Abreviaturas cortas
                    if alias in aff_lower or aff_lower in alias:
                        best_score = max(best_score, 0.9)

            # Fuzzy matching
            aff_normalized = _name_normalizer.remove_accents(aff_lower)
            snii_normalized = _name_normalizer.remove_accents(snii_institution)

            ratio = fuzz.ratio(snii_normalized, aff_normalized) / 100.0
            partial = fuzz.partial_ratio(snii_normalized, aff_normalized) / 100.0
            token = fuzz.token_sort_ratio(snii_normalized, aff_normalized) / 100.0

            score = max(ratio, partial * 0.9, token)
            best_score = max(best_score, score)

            # Tambien comparar contra aliases de SNII
            for alias in snii_aliases:
                alias_norm = _name_normalizer.remove_accents(alias)
                s = fuzz.token_sort_ratio(alias_norm, aff_normalized) / 100.0
                best_score = max(best_score, s)

        return best_score

    def merge_duplicates(
        self,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Fusiona candidatos duplicados (mismo ORCID desde ORCID y OpenAlex).

        Si un candidato ORCID y uno OpenAlex comparten el mismo ORCID iD,
        los fusiona tomando lo mejor de cada uno.

        Args:
            candidates: Lista con posibles duplicados.

        Returns:
            Lista sin duplicados, con datos fusionados.
        """
        by_orcid: dict[str, Candidate] = {}
        no_orcid: list[Candidate] = []

        for c in candidates:
            if c.orcid_id:
                if c.orcid_id in by_orcid:
                    existing = by_orcid[c.orcid_id]
                    # Merge: tomar lo mejor de cada uno
                    merged = Candidate(
                        source=CandidateSource.MERGED,
                        source_id=c.orcid_id,
                        given_name=existing.given_name or c.given_name,
                        family_name=existing.family_name or c.family_name,
                        affiliations=list(set(existing.affiliations + c.affiliations)),
                        works_count=max(existing.works_count, c.works_count),
                        cited_by_count=max(existing.cited_by_count, c.cited_by_count),
                        concepts=list(set(existing.concepts + c.concepts))[:15],
                        evidence=existing.evidence.combine(c.evidence),
                        orcid_id=c.orcid_id,
                        openalex_id=existing.openalex_id or c.openalex_id,
                    )
                    by_orcid[c.orcid_id] = merged
                else:
                    by_orcid[c.orcid_id] = c
            else:
                no_orcid.append(c)

        return list(by_orcid.values()) + no_orcid
