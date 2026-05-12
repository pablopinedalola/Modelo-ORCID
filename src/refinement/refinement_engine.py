"""
refinement_engine.py -- Motor de refinamiento iterativo de identidad.

Implementa el modelo matematico central del proyecto:

    d_{n+1}(v) = F(d_n(v))
    F(d)(v) = d(v) V  V_{u in N(v)} phi(d(u))

donde:
    v     = investigador SNII
    N(v)  = vecinos en el grafo academico
    phi   = funcion de extraccion de evidencia
    V     = operador de acumulacion (lattice join / max por dimension)

Cada iteracion:
    1. Para cada candidato, recorre sus vecinos en el grafo
    2. Extrae evidencia relevante via phi()
    3. Acumula evidencia via combine() (operador V)
    4. Recalcula confidence score
    5. Verifica convergencia

La identidad del investigador EMERGE progresivamente conforme el
sistema agrega evidencia y reduce incertidumbre.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from config import (
    MAX_ITERATIONS,
    CONVERGENCE_EPSILON,
    CONFIDENCE_THRESHOLD,
    EVIDENCE_WEIGHTS,
)
from src.models.schemas import (
    Candidate,
    EvidenceVector,
    NormalizedRecord,
    NodeType,
    EdgeType,
)
from src.graph.knowledge_graph import AcademicKnowledgeGraph
from src.normalizer.name_normalizer import NameNormalizer
from src.interpreter.evidence_trace import EvidenceTrace

logger = logging.getLogger(__name__)

_name_norm = NameNormalizer()


@dataclass
class IterationLog:
    """Registro de una iteracion del refinamiento.

    Attributes:
        iteration: Numero de iteracion (1-indexed).
        scores: Dict {candidate_id: confidence} tras la iteracion.
        max_delta: Maximo cambio absoluto respecto a la iteracion anterior.
        converged: Si se alcanzo convergencia en esta iteracion.
    """
    iteration: int
    scores: dict[str, float] = field(default_factory=dict)
    max_delta: float = 0.0
    converged: bool = False


@dataclass
class RefinementResult:
    """Resultado del proceso de refinamiento para un investigador.

    Attributes:
        researcher_id: ID del nodo SNII.
        researcher_name: Nombre del investigador.
        candidates: Candidatos con evidencia refinada.
        iterations: Numero de iteraciones ejecutadas.
        converged: Si el proceso convergio.
        history: Log de cada iteracion.
        best_candidate: Candidato con mayor confidence.
    """
    researcher_id: str
    researcher_name: str
    candidates: list[Candidate] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    history: list[IterationLog] = field(default_factory=list)
    trace: Optional[EvidenceTrace] = field(default=None, repr=False)

    @property
    def best_candidate(self) -> Optional[Candidate]:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.evidence.confidence())


class RefinementEngine:
    """Motor de refinamiento iterativo de identidad academica.

    Implementa F(d)(v) = d(v) V V_{u in N(v)} phi(d(u))

    El motor recorre iterativamente los vecinos de cada candidato
    en el grafo academico y acumula evidencia hasta convergencia.

    Attributes:
        graph: Knowledge graph academico.
        max_iterations: Maximo de iteraciones.
        epsilon: Umbral de convergencia.
        weights: Pesos de las dimensiones de evidencia.

    Examples:
        >>> engine = RefinementEngine(graph)
        >>> result = engine.refine(researcher_record, candidates)
        >>> print(f"Convergio en {result.iterations} iteraciones")
        >>> print(f"Mejor: {result.best_candidate.display_name}")
    """

    def __init__(
        self,
        graph: AcademicKnowledgeGraph,
        max_iterations: int = MAX_ITERATIONS,
        epsilon: float = CONVERGENCE_EPSILON,
        weights: Optional[dict[str, float]] = None,
        semantic_matcher=None,
    ) -> None:
        self.graph = graph
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.weights = weights or EVIDENCE_WEIGHTS
        self.semantic_matcher = semantic_matcher

    def _evidence_to_scores(self, ev: EvidenceVector) -> dict[str, float]:
        """Extract dimension scores from an EvidenceVector as dict."""
        return {
            "name": ev.name_score,
            "institution": ev.institution_score,
            "area": ev.area_score,
            "publication": ev.publication_score,
            "coauthor": ev.coauthor_score,
            "temporal": ev.temporal_score,
            "semantic": ev.semantic_score,
        }

    def refine(
        self,
        record: NormalizedRecord,
        candidates: list[Candidate],
    ) -> RefinementResult:
        """Ejecuta refinamiento iterativo para un investigador.

        Este es el loop principal del modelo matematico:

        for n = 1, 2, ...:
            for each candidate v:
                for each neighbor u in N(v):
                    evidence_new = phi(u)
                    d(v) = d(v) V evidence_new
            if converged: break

        Args:
            record: Investigador SNII normalizado.
            candidates: Candidatos con scores iniciales (de Fase 2).

        Returns:
            RefinementResult con candidatos refinados e historial.
        """
        researcher_id = f"snii:{record.id}"
        history: list[IterationLog] = []

        # == EVIDENCE TRACE (Fase 7: Interpretability) ==
        trace = EvidenceTrace(
            researcher_id=researcher_id,
            researcher_name=record.original.full_name,
        )

        if not candidates:
            trace.finalize(best_candidate_id=None, converged=True)
            return RefinementResult(
                researcher_id=researcher_id,
                researcher_name=record.original.full_name,
                candidates=[],
                iterations=0,
                converged=True,
                history=[],
                trace=trace,
            )

        # Mapear candidate_id -> Candidate para acceso rapido
        cand_map: dict[str, Candidate] = {}
        for c in candidates:
            cid = f"cand:{c.source.value}:{c.source_id}"
            cand_map[cid] = c

        # == SEMANTIC PRE-SCORING ==
        if self.semantic_matcher:
            logger.info(f"  Computing semantic scores...")
            for cand_id, candidate in cand_map.items():
                try:
                    # Gather paper titles from graph neighbors
                    paper_titles = []
                    neighbors = self.graph.get_neighbors(cand_id)
                    for n in neighbors:
                        if n["node_data"].get("type") == NodeType.PAPER.value:
                            title = n["node_data"].get("title", "")
                            if title:
                                paper_titles.append(title)

                    sem_score = self.semantic_matcher.compute_semantic_score(
                        record, candidate, paper_titles
                    )
                    candidate.evidence = candidate.evidence.combine(
                        EvidenceVector(semantic_score=sem_score)
                    )
                except Exception as e:
                    logger.debug(f"    Semantic error for {cand_id}: {e}")

        # Record initial states d_0(v) for all candidates
        for cand_id, candidate in cand_map.items():
            trace.record_initial_state(
                candidate_id=cand_id,
                display_name=candidate.display_name,
                scores=self._evidence_to_scores(candidate.evidence),
                confidence=candidate.evidence.confidence(self.weights),
            )

        logger.info(
            f"  Refinement: {len(candidates)} candidatos, "
            f"max_iter={self.max_iterations}, eps={self.epsilon}"
        )

        # == LOOP ITERATIVO ==
        for iteration in range(1, self.max_iterations + 1):
            trace.begin_iteration(iteration)

            prev_scores = {
                cid: c.evidence.confidence(self.weights)
                for cid, c in cand_map.items()
            }

            # Record states BEFORE iteration
            for cand_id, candidate in cand_map.items():
                trace.record_candidate_state_before(
                    candidate_id=cand_id,
                    display_name=candidate.display_name,
                    scores=self._evidence_to_scores(candidate.evidence),
                    confidence=candidate.evidence.confidence(self.weights),
                )

            # Para cada candidato: recorrer vecinos y acumular evidencia
            for cand_id, candidate in cand_map.items():
                neighbors = self.graph.get_neighbors(cand_id)

                for neighbor in neighbors:
                    # phi(u): extraer evidencia del vecino
                    new_evidence = self.phi(
                        neighbor,
                        record,
                        candidate,
                    )

                    if new_evidence is not None:
                        # Record contribution BEFORE combine
                        ev_before = self._evidence_to_scores(candidate.evidence)
                        ev_new = self._evidence_to_scores(new_evidence)

                        # V: acumular (max por dimension)
                        candidate.evidence = candidate.evidence.combine(
                            new_evidence
                        )

                        # Record each dimension's contribution
                        node_type = neighbor["node_data"].get("type", "")
                        edge_type = neighbor["edge_type"]
                        for dim in ["name", "institution", "area",
                                    "publication", "coauthor", "temporal",
                                    "semantic"]:
                            new_val = ev_new.get(dim, 0.0)
                            if new_val > 0:
                                trace.record_evidence_contribution(
                                    candidate_id=cand_id,
                                    neighbor_id=neighbor["node_id"],
                                    neighbor_type=node_type,
                                    edge_type=edge_type,
                                    dimension=dim,
                                    value=new_val,
                                    previous_value=ev_before.get(dim, 0.0),
                                    detail=(
                                        f"{node_type} '{neighbor['node_data'].get('label', '')[:40]}' "
                                        f"-> {dim}={new_val:.3f}"
                                    ),
                                )

            # Calcular scores actuales y delta
            curr_scores = {
                cid: c.evidence.confidence(self.weights)
                for cid, c in cand_map.items()
            }

            max_delta = 0.0
            for cid in cand_map:
                delta = abs(curr_scores[cid] - prev_scores[cid])
                max_delta = max(max_delta, delta)

            converged = max_delta < self.epsilon

            # Record states AFTER iteration
            for cand_id, candidate in cand_map.items():
                trace.record_candidate_state_after(
                    candidate_id=cand_id,
                    display_name=candidate.display_name,
                    scores=self._evidence_to_scores(candidate.evidence),
                    confidence=candidate.evidence.confidence(self.weights),
                )

            trace.end_iteration(converged=converged, max_delta=max_delta)

            log = IterationLog(
                iteration=iteration,
                scores={cid: round(s, 4) for cid, s in curr_scores.items()},
                max_delta=round(max_delta, 6),
                converged=converged,
            )
            history.append(log)

            logger.debug(
                f"    Iter {iteration}: max_delta={max_delta:.6f} "
                f"{'[CONVERGED]' if converged else ''}"
            )

            if converged:
                break

        # Ordenar candidatos por confidence final
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.evidence.confidence(self.weights),
            reverse=True,
        )

        # Finalize trace
        best_cid = None
        if sorted_candidates:
            bc = sorted_candidates[0]
            best_cid = f"cand:{bc.source.value}:{bc.source_id}"
        trace.finalize(
            best_candidate_id=best_cid,
            converged=history[-1].converged if history else True,
        )

        result = RefinementResult(
            researcher_id=researcher_id,
            researcher_name=record.original.full_name,
            candidates=sorted_candidates,
            iterations=len(history),
            converged=history[-1].converged if history else True,
            history=history,
            trace=trace,
        )

        logger.info(
            f"  Refinement: {'convergio' if result.converged else 'NO convergio'} "
            f"en {result.iterations} iteraciones"
        )

        return result

    def phi(
        self,
        neighbor: dict,
        record: NormalizedRecord,
        candidate: Candidate,
    ) -> Optional[EvidenceVector]:
        """Funcion phi -- extrae evidencia de un vecino del grafo.

        phi es la funcion de extraccion que transforma la informacion
        de un vecino en evidencia para la desambiguacion.

        Interpretacion por tipo de vecino:
            - PAPER       -> publication_score, temporal_score
            - INSTITUTION -> institution_score
            - DISCIPLINE  -> area_score
            - CANDIDATE   -> coauthor_score (si es coautor)

        Args:
            neighbor: Dict con node_id, node_data, edge_type.
            record: Investigador SNII de referencia.
            candidate: Candidato que se esta evaluando.

        Returns:
            EvidenceVector parcial, o None si no aporta evidencia.
        """
        node_data = neighbor["node_data"]
        node_type = node_data.get("type", "")
        edge_type = neighbor["edge_type"]

        # ── PAPER -> publication_score, temporal_score ──
        if node_type == NodeType.PAPER.value:
            return self._phi_paper(node_data, record)

        # ── INSTITUTION -> institution_score ──
        if node_type == NodeType.INSTITUTION.value:
            return self._phi_institution(node_data, record)

        # ── DISCIPLINE -> area_score ──
        if node_type == NodeType.DISCIPLINE.value:
            return self._phi_discipline(node_data, record)

        # ── CANDIDATE (coautor) -> coauthor_score ──
        if (node_type == NodeType.CANDIDATE.value
                and edge_type == EdgeType.COAUTHOR.value):
            return self._phi_coauthor(neighbor, record)

        return None

    def _phi_paper(
        self,
        paper_data: dict,
        record: NormalizedRecord,
    ) -> EvidenceVector:
        """Extrae evidencia de una publicacion.

        Heuristicas:
        - Publicaciones existentes aumentan publication_score
        - cited_by_count > 0 da mayor peso
        - Anio de publicacion razonable (1990-2030) da temporal_score

        Args:
            paper_data: Datos del nodo paper.
            record: Investigador SNII.

        Returns:
            EvidenceVector con publication_score y temporal_score.
        """
        pub_score = 0.0
        temp_score = 0.0

        # Existe publicacion -> evidencia positiva base
        title = paper_data.get("title", "")
        if title:
            pub_score = 0.3  # Base por existencia

        # Publicacion citada -> evidencia mas fuerte
        cited = paper_data.get("cited_by", 0) or 0
        if cited > 0:
            pub_score = min(0.3 + 0.1 * (cited ** 0.3), 0.9)

        # Temporalidad razonable
        year = paper_data.get("year")
        if year and isinstance(year, (int, float)):
            year = int(year)
            if 1990 <= year <= 2030:
                temp_score = 0.5
                if 2010 <= year <= 2026:
                    temp_score = 0.7  # Mas reciente = mas relevante

        return EvidenceVector(
            publication_score=pub_score,
            temporal_score=temp_score,
        )

    def _phi_institution(
        self,
        inst_data: dict,
        record: NormalizedRecord,
    ) -> EvidenceVector:
        """Extrae evidencia de una institucion vecina.

        Compara la institucion del candidato con la del investigador SNII.

        Args:
            inst_data: Datos del nodo institucion.
            record: Investigador SNII.

        Returns:
            EvidenceVector con institution_score.
        """
        inst_label = inst_data.get("label", "")
        inst_ror = inst_data.get("ror_id", "")
        snii_ror = record.ror_id

        # Match por ROR ID (exacto)
        if inst_ror and snii_ror and inst_ror == snii_ror:
            return EvidenceVector(institution_score=1.0)

        # Match por nombre fuzzy
        if inst_label and record.normalized_institution:
            inst_norm = _name_norm.remove_accents(inst_label.lower())
            snii_norm = _name_norm.remove_accents(
                record.normalized_institution.lower()
            )
            ratio = fuzz.token_sort_ratio(inst_norm, snii_norm) / 100.0

            # Tambien comparar contra aliases
            for alias in record.institution_aliases:
                alias_norm = _name_norm.remove_accents(alias.lower())
                s = fuzz.ratio(alias_norm, inst_norm) / 100.0
                ratio = max(ratio, s)

            if ratio > 0.5:
                return EvidenceVector(institution_score=ratio)

        return EvidenceVector()

    def _phi_discipline(
        self,
        disc_data: dict,
        record: NormalizedRecord,
    ) -> EvidenceVector:
        """Extrae evidencia de una disciplina/concepto vecino.

        Compara la disciplina del candidato con la del investigador SNII.

        Args:
            disc_data: Datos del nodo disciplina.
            record: Investigador SNII.

        Returns:
            EvidenceVector con area_score.
        """
        disc_label = disc_data.get("label", "")
        disc_area = disc_data.get("area", "")

        snii_disc = record.original.disciplina
        snii_area = record.original.area

        # Match por area CONAHCyT
        if disc_area and snii_area and disc_area == snii_area:
            return EvidenceVector(area_score=0.6)

        # Match por nombre de disciplina (fuzzy)
        if disc_label and snii_disc:
            disc_norm = _name_norm.remove_accents(disc_label.lower())
            snii_norm = _name_norm.remove_accents(snii_disc.lower())

            ratio = fuzz.token_sort_ratio(disc_norm, snii_norm) / 100.0
            partial = fuzz.partial_ratio(disc_norm, snii_norm) / 100.0
            score = max(ratio, partial * 0.8)

            if score > 0.3:
                return EvidenceVector(area_score=min(score, 0.9))

        return EvidenceVector()

    def _phi_coauthor(
        self,
        neighbor: dict,
        record: NormalizedRecord,
    ) -> EvidenceVector:
        """Extrae evidencia de un coautor.

        Si un coautor del candidato tambien esta afiliado a la misma
        institucion que el investigador SNII, eso es evidencia positiva.

        Args:
            neighbor: Vecino de tipo CANDIDATE con edge COAUTHOR.
            record: Investigador SNII.

        Returns:
            EvidenceVector con coauthor_score.
        """
        coauthor_id = neighbor["node_id"]

        # Verificar si el coautor tiene afiliacion con la misma institucion
        coauthor_neighbors = self.graph.get_neighbors(coauthor_id)
        for cn in coauthor_neighbors:
            cn_type = cn["node_data"].get("type", "")
            if cn_type == NodeType.INSTITUTION.value:
                cn_ror = cn["node_data"].get("ror_id", "")
                if cn_ror and record.ror_id and cn_ror == record.ror_id:
                    return EvidenceVector(coauthor_score=0.7)

                # Fuzzy match de institucion
                cn_label = cn["node_data"].get("label", "")
                if cn_label and record.normalized_institution:
                    ratio = fuzz.token_sort_ratio(
                        _name_norm.remove_accents(cn_label.lower()),
                        _name_norm.remove_accents(
                            record.normalized_institution.lower()
                        ),
                    ) / 100.0
                    if ratio > 0.7:
                        return EvidenceVector(coauthor_score=0.5)

        # Coautor existe pero sin match de institucion
        return EvidenceVector(coauthor_score=0.1)

    def check_convergence(
        self,
        prev_scores: dict[str, float],
        curr_scores: dict[str, float],
    ) -> bool:
        """Verifica convergencia: max|conf_new - conf_old| < epsilon.

        Args:
            prev_scores: Scores de la iteracion anterior.
            curr_scores: Scores de la iteracion actual.

        Returns:
            True si convergio.
        """
        if not prev_scores or not curr_scores:
            return True

        max_delta = max(
            abs(curr_scores.get(k, 0) - prev_scores.get(k, 0))
            for k in set(prev_scores) | set(curr_scores)
        )
        return max_delta < self.epsilon
