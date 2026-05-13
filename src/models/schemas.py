"""
schemas.py — Modelos de datos centrales del sistema Modelo-ORCID.

Define las dataclasses tipadas que fluyen por todo el pipeline:
    SNIIRecord → NormalizedRecord → Candidate → MatchResult → ResearcherProfile

Cada modelo incluye factory methods, validación y serialización JSON.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES
# ═══════════════════════════════════════════════════════════════════════════════

class SNIILevel(str, Enum):
    """Niveles del Sistema Nacional de Investigadores."""
    CANDIDATO = "C"
    SNI_1 = "1"
    SNI_2 = "2"
    SNI_3 = "3"
    EMERITO = "E"
    UNKNOWN = "?"

    @classmethod
    def from_string(cls, value: str) -> "SNIILevel":
        """Parsea un string flexible al nivel SNII correspondiente.

        Args:
            value: String como 'SNI I', 'Nivel 2', 'C', 'Candidato', etc.

        Returns:
            SNIILevel correspondiente.

        Examples:
            >>> SNIILevel.from_string("SNI I")
            <SNIILevel.SNI_1: '1'>
            >>> SNIILevel.from_string("Candidato")
            <SNIILevel.CANDIDATO: 'C'>
        """
        if not value:
            return cls.UNKNOWN

        normalized = value.strip().upper()

        level_map = {
            "C": cls.CANDIDATO, "CANDIDATO": cls.CANDIDATO, "SNI C": cls.CANDIDATO,
            "1": cls.SNI_1, "I": cls.SNI_1, "SNI 1": cls.SNI_1, "SNI I": cls.SNI_1,
            "NIVEL 1": cls.SNI_1, "NIVEL I": cls.SNI_1,
            "2": cls.SNI_2, "II": cls.SNI_2, "SNI 2": cls.SNI_2, "SNI II": cls.SNI_2,
            "NIVEL 2": cls.SNI_2, "NIVEL II": cls.SNI_2,
            "3": cls.SNI_3, "III": cls.SNI_3, "SNI 3": cls.SNI_3, "SNI III": cls.SNI_3,
            "NIVEL 3": cls.SNI_3, "NIVEL III": cls.SNI_3,
            "E": cls.EMERITO, "EMERITO": cls.EMERITO, "EMÉRITO": cls.EMERITO,
            "SNI E": cls.EMERITO,
        }

        return level_map.get(normalized, cls.UNKNOWN)


class CandidateSource(str, Enum):
    """Fuente de un candidato."""
    ORCID = "orcid"
    OPENALEX = "openalex"
    MERGED = "merged"


class Verdict(str, Enum):
    """Veredicto del Semantic Judge."""
    COMPATIBLE = "compatible"
    AMBIGUOUS = "ambiguous"
    INCOMPATIBLE = "incompatible"


class NodeType(str, Enum):
    """Tipos de nodo en el Knowledge Graph."""
    RESEARCHER = "researcher"
    CANDIDATE = "candidate"
    PAPER = "paper"
    INSTITUTION = "institution"
    DISCIPLINE = "discipline"
    ORCID = "orcid"
    TOPIC = "topic"
    CONCEPT = "concept"
    VENUE = "venue"
    CITATION = "citation"


class EdgeType(str, Enum):
    """Tipos de relación en el Knowledge Graph."""
    AUTHORED = "authored"
    AFFILIATED_WITH = "affiliated_with"
    COAUTHOR = "coauthor"
    BELONGS_TO_AREA = "belongs_to_area"
    HAS_ORCID = "has_orcid"
    CANDIDATE_FOR = "candidate_for"
    CITES = "cites"
    RELATED_TO_TOPIC = "related_to_topic"
    PUBLISHED_IN = "published_in"


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRO SNII (dato crudo del padrón)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SNIIRecord:
    """Registro crudo de un investigador del padrón SNII.

    Representa los campos tal como aparecen en el CSV/Excel original,
    antes de cualquier normalización.

    Attributes:
        id: Identificador único interno (UUID generado automáticamente).
        nombre: Nombre(s) de pila del investigador.
        paterno: Apellido paterno.
        materno: Apellido materno (puede estar vacío).
        institucion: Nombre de la institución de adscripción.
        dependencia: Dependencia dentro de la institución.
        subdependencia: Subdependencia (puede estar vacía).
        area: Área de conocimiento del CONAHCyT (I-VII).
        disciplina: Disciplina específica del investigador.
        nivel: Nivel SNII (Candidato, I, II, III, Emérito).

    Examples:
        >>> record = SNIIRecord(
        ...     nombre="CARLOS ALBERTO",
        ...     paterno="GARCIA",
        ...     materno="LOPEZ",
        ...     institucion="UNIVERSIDAD NACIONAL AUTONOMA DE MEXICO",
        ...     dependencia="INSTITUTO DE FISICA",
        ...     area="I",
        ...     disciplina="FISICA DE PARTICULAS",
        ...     nivel=SNIILevel.SNI_2
        ... )
        >>> record.full_name
        'CARLOS ALBERTO GARCIA LOPEZ'
    """

    nombre: str
    paterno: str
    materno: str = ""
    institucion: str = ""
    dependencia: str = ""
    subdependencia: str = ""
    area: str = ""
    disciplina: str = ""
    nivel: SNIILevel = SNIILevel.UNKNOWN
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def full_name(self) -> str:
        """Nombre completo concatenado (nombre + paterno + materno)."""
        parts = [self.nombre, self.paterno, self.materno]
        return " ".join(p for p in parts if p).strip()

    @property
    def area_label(self) -> str:
        """Etiqueta legible del área CONAHCyT."""
        area_labels = {
            "I": "Físico-Matemáticas y Ciencias de la Tierra",
            "II": "Biología y Química",
            "III": "Ciencias Médicas y de la Salud",
            "IV": "Humanidades y Ciencias de la Conducta",
            "V": "Ciencias Sociales",
            "VI": "Biotecnología y Ciencias Agropecuarias",
            "VII": "Ingenierías",
        }
        return area_labels.get(self.area.strip().upper(), self.area)

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        d = asdict(self)
        d["full_name"] = self.full_name
        d["area_label"] = self.area_label
        d["nivel"] = self.nivel.value
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRO NORMALIZADO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NormalizedRecord:
    """Investigador SNII tras normalización de nombre e institución.

    Extiende el registro original con aliases de nombre, forma canónica
    de la institución, y tokens para matching.

    Attributes:
        original: Registro SNII original sin modificar.
        normalized_name: Nombre en forma normalizada (lowercase, sin acentos).
        name_aliases: Lista de variaciones del nombre para búsqueda.
        normalized_institution: Institución en forma canónica.
        institution_aliases: Lista de variaciones de la institución.
        name_tokens: Tokens del nombre para fuzzy matching.
        ror_id: ROR ID de la institución (si se resolvió).

    Examples:
        >>> normalized = NormalizedRecord(
        ...     original=record,
        ...     normalized_name="carlos alberto garcia lopez",
        ...     name_aliases=["C. Garcia Lopez", "Garcia-Lopez, Carlos"],
        ...     normalized_institution="Universidad Nacional Autónoma de México",
        ...     institution_aliases=["UNAM"],
        ... )
    """

    original: SNIIRecord
    normalized_name: str = ""
    name_aliases: list[str] = field(default_factory=list)
    normalized_institution: str = ""
    institution_aliases: list[str] = field(default_factory=list)
    name_tokens: set[str] = field(default_factory=set)
    ror_id: Optional[str] = None

    @property
    def id(self) -> str:
        """ID del registro original."""
        return self.original.id

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            "id": self.id,
            "original": self.original.to_dict(),
            "normalized_name": self.normalized_name,
            "name_aliases": self.name_aliases,
            "normalized_institution": self.normalized_institution,
            "institution_aliases": self.institution_aliases,
            "name_tokens": list(self.name_tokens),
            "ror_id": self.ror_id,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR DE EVIDENCIA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceVector:
    """Vector multi-dimensional de evidencia para desambiguación.

    Cada dimensión captura un tipo diferente de evidencia que contribuye
    a la confianza de que un candidato corresponde a un investigador SNII.

    Los valores están en el rango [0.0, 1.0].

    El confidence score final es una combinación ponderada de todas
    las dimensiones según los pesos definidos en config.EVIDENCE_WEIGHTS.

    Attributes:
        name_score: Similitud de nombre (fuzzy + embeddings).
        institution_score: Coincidencia institucional (ROR-backed).
        area_score: Superposición de área/disciplina.
        publication_score: Evidencia basada en publicaciones.
        coauthor_score: Evidencia de red de coautoría.
        temporal_score: Consistencia temporal de la carrera.
        semantic_score: Score del Semantic Judge (LLM).
    """

    name_score: float = 0.0
    institution_score: float = 0.0
    area_score: float = 0.0
    publication_score: float = 0.0
    coauthor_score: float = 0.0
    temporal_score: float = 0.0
    semantic_score: float = 0.0

    def confidence(self, weights: Optional[dict[str, float]] = None) -> float:
        """Calcula el score de confianza ponderado.

        Args:
            weights: Diccionario {dimension: peso}. Si es None, usa pesos
                     de config.EVIDENCE_WEIGHTS.

        Returns:
            Score de confianza en [0.0, 1.0].

        Examples:
            >>> ev = EvidenceVector(name_score=0.9, institution_score=0.8)
            >>> ev.confidence({"name": 0.5, "institution": 0.5,
            ...     "area": 0, "publication": 0, "coauthor": 0,
            ...     "temporal": 0, "semantic": 0})
            0.85
        """
        if weights is None:
            from config import EVIDENCE_WEIGHTS
            weights = EVIDENCE_WEIGHTS

        score_map = {
            "name": self.name_score,
            "institution": self.institution_score,
            "area": self.area_score,
            "publication": self.publication_score,
            "coauthor": self.coauthor_score,
            "temporal": self.temporal_score,
            "semantic": self.semantic_score,
        }

        total = sum(
            score_map.get(dim, 0.0) * w for dim, w in weights.items()
        )
        weight_sum = sum(weights.values())
        return total / weight_sum if weight_sum > 0 else 0.0

    def combine(self, other: "EvidenceVector") -> "EvidenceVector":
        """Operador ∨ — combina dos vectores tomando el máximo por dimensión.

        Implementa la semántica de acumulación del modelo matemático:
        la evidencia solo crece, nunca se pierde.

        Args:
            other: Otro vector de evidencia.

        Returns:
            Nuevo EvidenceVector con max(self, other) por dimensión.
        """
        return EvidenceVector(
            name_score=max(self.name_score, other.name_score),
            institution_score=max(self.institution_score, other.institution_score),
            area_score=max(self.area_score, other.area_score),
            publication_score=max(self.publication_score, other.publication_score),
            coauthor_score=max(self.coauthor_score, other.coauthor_score),
            temporal_score=max(self.temporal_score, other.temporal_score),
            semantic_score=max(self.semantic_score, other.semantic_score),
        )

    def distance(self, other: "EvidenceVector") -> float:
        """Distancia máxima absoluta entre dos vectores.

        Útil para verificar convergencia del refinamiento iterativo.

        Args:
            other: Otro vector de evidencia.

        Returns:
            max(|self.dim - other.dim|) para todas las dimensiones.
        """
        return max(
            abs(self.name_score - other.name_score),
            abs(self.institution_score - other.institution_score),
            abs(self.area_score - other.area_score),
            abs(self.publication_score - other.publication_score),
            abs(self.coauthor_score - other.coauthor_score),
            abs(self.temporal_score - other.temporal_score),
            abs(self.semantic_score - other.semantic_score),
        )

    def to_dict(self) -> dict:
        """Serializa a diccionario con confidence incluido."""
        d = asdict(self)
        d["confidence"] = round(self.confidence(), 4)
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# CANDIDATO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    """Candidato ORCID/OpenAlex para un investigador SNII.

    Attributes:
        source: Fuente del candidato (ORCID, OpenAlex, merged).
        source_id: ID en la fuente (ORCID iD o OpenAlex ID).
        given_name: Nombre(s) del candidato.
        family_name: Apellido(s) del candidato.
        affiliations: Lista de afiliaciones conocidas.
        works_count: Número de publicaciones.
        cited_by_count: Número de citas recibidas.
        concepts: Lista de temas/conceptos de investigación.
        evidence: Vector de evidencia acumulada.
        orcid_id: ORCID iD si está disponible.
        openalex_id: OpenAlex ID si está disponible.
    """

    source: CandidateSource
    source_id: str
    given_name: str = ""
    family_name: str = ""
    affiliations: list[str] = field(default_factory=list)
    works_count: int = 0
    cited_by_count: int = 0
    concepts: list[str] = field(default_factory=list)
    evidence: EvidenceVector = field(default_factory=EvidenceVector)
    orcid_id: Optional[str] = None
    openalex_id: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Nombre completo para visualización."""
        parts = [self.given_name, self.family_name]
        return " ".join(p for p in parts if p).strip()

    @property
    def confidence(self) -> float:
        """Score de confianza del evidence vector."""
        return self.evidence.confidence()

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        d = asdict(self)
        d["display_name"] = self.display_name
        d["confidence"] = round(self.confidence, 4)
        d["source"] = self.source.value
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTADO DE RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """Resultado completo de la búsqueda de candidatos para un investigador.

    Encapsula todos los candidatos encontrados desde todas las fuentes
    (ORCID, OpenAlex) junto con metadatos de la búsqueda.

    Attributes:
        snii_id: ID del registro SNII.
        snii_name: Nombre completo del investigador SNII.
        candidates: Lista de candidatos ordenados por score.
        orcid_candidates_count: Candidatos encontrados vía ORCID.
        openalex_candidates_count: Candidatos encontrados vía OpenAlex.
        best_candidate: Mejor candidato (mayor confidence), o None.
        search_time_seconds: Tiempo total de búsqueda.
        errors: Lista de errores no fatales durante la búsqueda.
    """

    snii_id: str
    snii_name: str
    candidates: list[Candidate] = field(default_factory=list)
    orcid_candidates_count: int = 0
    openalex_candidates_count: int = 0
    search_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def best_candidate(self) -> Optional[Candidate]:
        """Candidato con mayor confidence score."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.confidence)

    @property
    def total_candidates(self) -> int:
        """Total de candidatos encontrados."""
        return len(self.candidates)

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        best = self.best_candidate
        return {
            "snii_id": self.snii_id,
            "snii_name": self.snii_name,
            "total_candidates": self.total_candidates,
            "orcid_candidates": self.orcid_candidates_count,
            "openalex_candidates": self.openalex_candidates_count,
            "best_candidate": best.to_dict() if best else None,
            "search_time_seconds": round(self.search_time_seconds, 3),
            "errors": self.errors,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTADO DE MATCH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatchResult:
    """Resultado final de desambiguación para un investigador.

    Attributes:
        snii_id: ID del registro SNII.
        snii_name: Nombre completo del investigador SNII.
        orcid_id: ORCID iD asignado (puede ser None si no se encontró).
        openalex_id: OpenAlex ID asignado.
        confidence: Score de confianza final [0.0, 1.0].
        validated: True si confidence > threshold.
        verdict: Veredicto del Semantic Judge.
        explanation: Explicación del veredicto.
        candidates_evaluated: Número total de candidatos evaluados.
        iterations: Número de iteraciones de refinamiento.
        timestamp: Momento de la resolución.
    """

    snii_id: str
    snii_name: str
    orcid_id: Optional[str] = None
    openalex_id: Optional[str] = None
    confidence: float = 0.0
    validated: bool = False
    verdict: Verdict = Verdict.AMBIGUOUS
    explanation: str = ""
    candidates_evaluated: int = 0
    iterations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def to_json(self) -> str:
        """Serializa a JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFIL DE INVESTIGADOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResearcherProfile:
    """Perfil enriquecido tipo 'Wikipedia académica' de un investigador.

    Combina datos del SNII, ORCID, OpenAlex y ROR para crear un perfil
    navegable y completo.

    Attributes:
        id: Identificador único del perfil.
        full_name: Nombre completo del investigador.
        snii_level: Nivel SNII.
        institution: Institución de adscripción.
        department: Dependencia / departamento.
        area: Área de conocimiento.
        discipline: Disciplina específica.
        orcid_id: ORCID iD (si se resolvió).
        openalex_id: OpenAlex ID.
        ror_id: ROR ID de la institución.
        confidence: Score de confianza de la resolución.
        validated: Si la identidad fue validada.
        publications: Lista de publicaciones relevantes.
        coauthors: Lista de coautores frecuentes.
        concepts: Temas/conceptos de investigación.
        h_index: Índice h (de OpenAlex).
        works_count: Total de publicaciones.
        cited_by_count: Total de citas.
        external_links: Links a ORCID, OpenAlex, Scholar, etc.
    """

    id: str
    full_name: str
    snii_level: str = ""
    institution: str = ""
    department: str = ""
    area: str = ""
    discipline: str = ""
    orcid_id: Optional[str] = None
    openalex_id: Optional[str] = None
    ror_id: Optional[str] = None
    confidence: float = 0.0
    validated: bool = False
    publications: list[dict] = field(default_factory=list)
    coauthors: list[dict] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    h_index: int = 0
    works_count: int = 0
    cited_by_count: int = 0
    external_links: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return asdict(self)

    def to_json(self) -> str:
        """Serializa a JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
