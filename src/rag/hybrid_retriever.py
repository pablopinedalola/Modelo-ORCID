"""
hybrid_retriever.py — Motor de retrieval híbrido para perfiles SNII.

Combina:
    1. Dense semantic retrieval (FAISS + embeddings)
    2. Sparse lexical retrieval (BM25)
    3. Metadata filtering (institución, área, disciplina, nivel)
    4. Query understanding (detección de keywords académicas)

El score final es una fusión ponderada configurable:
    final_score = α * semantic_score + (1 - α) * lexical_score

Con α = 0.65 por defecto (65% semántico, 35% léxico).

Incluye explicabilidad básica: cada resultado explica por qué apareció.

Examples:
    >>> retriever = HybridRetriever()
    >>> retriever.load()
    >>> results = retriever.search("redes complejas UNAM", top_k=10)
    >>> for r in results:
    ...     print(f"{r['nombre_completo']} — {r['score']:.3f} ({r['search_method']})")
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from src.rag.query_interpreter import QueryInterpreter
from config import BASE_DIR, PROCESSED_DATA_DIR, HYBRID_SEMANTIC_WEIGHT, HYBRID_LEXICAL_WEIGHT

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = BASE_DIR / "data" / "vector_store"

# ─── Pesos por defecto ───────────────────────────────────────────────
DEFAULT_SEMANTIC_WEIGHT = HYBRID_SEMANTIC_WEIGHT
DEFAULT_LEXICAL_WEIGHT = HYBRID_LEXICAL_WEIGHT

# ─── Instituciones conocidas (para query understanding) ──────────────
INSTITUTION_ALIASES = {
    "unam": "UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO",
    "ipn": "INSTITUTO POLITÉCNICO NACIONAL",
    "cinvestav": "CENTRO DE INVESTIGACIÓN Y DE ESTUDIOS AVANZADOS",
    "uam": "UNIVERSIDAD AUTÓNOMA METROPOLITANA",
    "udg": "UNIVERSIDAD DE GUADALAJARA",
    "buap": "BENEMÉRITA UNIVERSIDAD AUTÓNOMA DE PUEBLA",
    "uanl": "UNIVERSIDAD AUTÓNOMA DE NUEVO LEÓN",
    "uaem": "UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MORELOS",
    "cicese": "CENTRO DE INVESTIGACIÓN CIENTÍFICA Y DE EDUCACIÓN SUPERIOR DE ENSENADA",
    "uaslp": "UNIVERSIDAD AUTÓNOMA DE SAN LUIS POTOSÍ",
    "uv": "UNIVERSIDAD VERACRUZANA",
    "colmex": "EL COLEGIO DE MÉXICO",
    "uaq": "UNIVERSIDAD AUTÓNOMA DE QUERÉTARO",
    "tec": "TECNOLÓGICO DE MONTERREY",
    "itesm": "TECNOLÓGICO DE MONTERREY",
    "uag": "UNIVERSIDAD AUTÓNOMA DE GUADALAJARA",
}

# ─── Áreas CONAHCyT ─────────────────────────────────────────────────
AREA_KEYWORDS = {
    "I": ["fisica", "matematicas", "tierra", "geofisica", "astronomia"],
    "II": ["biologia", "quimica", "bioquimica"],
    "III": ["medicina", "salud", "epidemiologia", "farmacologia"],
    "IV": ["humanidades", "conducta", "psicologia", "filosofia", "linguistica"],
    "V": ["sociales", "economia", "derecho", "politica", "sociologia"],
    "VI": ["biotecnologia", "agropecuarias", "agronomia", "veterinaria"],
    "VII": ["ingenieria", "computacion", "mecanica", "electronica", "mecatronica"],
}

# ─── Niveles SNI ─────────────────────────────────────────────────────
LEVEL_KEYWORDS = {
    "C": ["candidato"],
    "1": ["sni 1", "sni i", "nivel 1", "nivel i"],
    "2": ["sni 2", "sni ii", "nivel 2", "nivel ii"],
    "3": ["sni 3", "sni iii", "nivel 3", "nivel iii"],
    "E": ["emerito"],
}


class QueryAnalysis:
    """Resultado del análisis de una consulta académica."""

    def __init__(self) -> None:
        self.raw_query: str = ""
        self.clean_query: str = ""
        self.institution_filter: Optional[str] = None
        self.institution_alias: Optional[str] = None
        self.area_filter: Optional[str] = None
        self.level_filter: Optional[str] = None
        self.discipline_keywords: list[str] = []
        self.detected_features: list[str] = []

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "clean_query": self.clean_query,
            "institution_filter": self.institution_filter,
            "institution_alias": self.institution_alias,
            "area_filter": self.area_filter,
            "level_filter": self.level_filter,
            "discipline_keywords": self.discipline_keywords,
            "detected_features": self.detected_features,
        }


class HybridRetriever:
    """Motor de retrieval híbrido (semántico + léxico + metadata).

    Combina FAISS (dense) con BM25 (sparse) y filtrado por metadata
    para búsqueda académica de alta precisión.

    Attributes:
        semantic_weight: Peso del score semántico (0-1).
        lexical_weight: Peso del score léxico (0-1).
        faiss_store: Índice FAISS.
        embedding_pipeline: Pipeline de embeddings.
        bm25_retriever: Retriever BM25.
        profiles: Perfiles completos.
    """

    def __init__(
        self,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        store_dir: Optional[Path] = None,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR

        self.faiss_store = None
        self.embedding_pipeline = None
        self.bm25_retriever = None
        self.profiles: list[dict] = []
        self._profiles_by_id: dict[str, dict] = {}
        self._semantic_ready = False
        self._lexical_ready = False
        self.interpreter = QueryInterpreter()

    def load(self) -> bool:
        """Carga todos los componentes del retriever híbrido.

        Returns:
            True si al menos un método de búsqueda está disponible.
        """
        # 1. Perfiles UNAM (Exclusivo)
        self.profiles = []
        unam_path = PROCESSED_DATA_DIR / "unam_authors.json"
        if unam_path.exists():
            with open(unam_path, "r", encoding="utf-8") as f:
                self.profiles.extend(json.load(f))
                
        if self.profiles:
            self._profiles_by_id = {p["id"]: p for p in self.profiles}
            logger.info(f"  📋 {len(self.profiles)} perfiles UNAM cargados")
        else:
            logger.warning(f"  ⚠️  No se encontró el corpus UNAM")
            return False

        # 2. FAISS semantic (Opcional en modo local, intentamos cargar)
        try:
            from src.rag.faiss_store import FAISSStore
            from src.rag.embedding_pipeline import EmbeddingPipeline

            self.faiss_store = FAISSStore(store_dir=self.store_dir)
            if self.faiss_store.load():
                self.embedding_pipeline = EmbeddingPipeline(store_dir=self.store_dir)
                self._semantic_ready = True
                logger.info("  ✅ Retrieval semántico (FAISS) listo")
        except Exception as e:
            logger.warning(f"  ⚠️  FAISS no disponible: {e}")

        # 3. BM25 lexical (Construcción forzada para el corpus local UNAM)
        try:
            from src.rag.bm25_retriever import BM25Retriever
            self.bm25_retriever = BM25Retriever(store_dir=self.store_dir)
            
            logger.info("  🔧 Construyendo índice BM25 desde corpus UNAM local...")
            self.bm25_retriever.build_index(self.profiles)
            self._lexical_ready = True
            logger.info("  ✅ BM25 listo para corpus local")
        except Exception as e:
            logger.warning(f"  ⚠️  BM25 no disponible: {e}")

        ready = self._semantic_ready or self._lexical_ready
        method = self._get_search_method()
        logger.info(f"  🔍 Método de búsqueda: {method}")
        return ready

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[dict] = None,
        include_explanation: bool = True,
    ) -> list[dict]:
        """Búsqueda híbrida: semántica + léxica + metadata.

        Args:
            query: Consulta en lenguaje natural.
            top_k: Máximo de resultados.
            min_score: Score mínimo.
            filters: Filtros de metadata opcionales:
                     {"institucion": "...", "area": "...",
                      "disciplina": "...", "nivel": "..."}.
            include_explanation: Si incluir explicación por resultado.

        Returns:
            Lista de resultados con score híbrido + explicación.
        """
        if not query or not query.strip():
            return []

        query = query.strip()

        # 1. Analizar consulta con Hybrid Analysis (metadata)
        analysis = self.analyze_query(query)

        # 1.5 Analizar con QueryInterpreter (semantic expansions)
        interp = self.interpreter.interpret(query)
        expanded_query = interp["rewritten_query"]
        if interp["institutions"]:
            analysis.institution_filter = interp["institutions"][0]
            analysis.detected_features.append(f"institución (QA): {interp['institutions'][0].upper()}")

        print("\n" + "="*50)
        print("🔍 RUNTIME LOG: QUERY INTERPRETER")
        print("="*50)
        print(f"[LOG] Original Query: '{query}'")
        print(f"[LOG] Normalized Query: '{interp.get('normalized_query', query.lower())}'")
        print(f"[LOG] Institution Detected: {interp['institutions']}")
        print(f"[LOG] Semantic Expansions: {interp['expanded_concepts']}")
        print(f"[LOG] Expanded Query (Dense): '{expanded_query}'")
        print("="*50 + "\n")

        # 2. Merge filters from query analysis + explicit filters
        merged_filters = self._merge_filters(analysis, filters)

        # 3. Retrieval multi-señal
        semantic_results = {}
        lexical_results = {}

        search_query = expanded_query or analysis.clean_query or query

        print(f"\n[LOG] Top-k Retrieval RAW:")

        # Semántico
        if self._semantic_ready:
            sem = self._semantic_search(search_query, top_k=top_k * 2)
            semantic_results = {r["profile_id"]: r for r in sem}

        # Léxico
        if self._lexical_ready:
            lex = self._lexical_search(query, top_k=top_k * 2)
            lexical_results = {r["profile_id"]: r for r in lex}

        # 4. Fusión
        fused = self._fuse_results(semantic_results, lexical_results)

        # 5. Metadata filtering
        if merged_filters:
            fused = self._apply_filters(fused, merged_filters)

        # 6. Score boost por query features
        fused = self._apply_boosts(fused, analysis)

        # 7. Sort + top-k
        fused.sort(key=lambda x: x["score"], reverse=True)
        fused = [r for r in fused if r["score"] >= min_score][:top_k]

        # 8. Enrich + explain
        results = []
        for rank, r in enumerate(fused, 1):
            enriched = self._enrich_result(r, rank, analysis, include_explanation)
            results.append(enriched)
            
        print(f"[LOG] Results AFTER institution filtering: {len(results)} autores válidos")

        return results

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analiza una consulta académica para extraer intenciones.

        Detecta:
            - Instituciones mencionadas (UNAM, IPN, etc.)
            - Áreas CONAHCyT
            - Niveles SNI
            - Keywords disciplinares

        Args:
            query: Consulta original.

        Returns:
            QueryAnalysis con features detectadas.
        """
        analysis = QueryAnalysis()
        analysis.raw_query = query

        q_lower = self._normalize(query)
        q_tokens = set(q_lower.split())
        remaining_tokens = list(q_lower.split())

        # Detectar institución
        for alias, full_name in INSTITUTION_ALIASES.items():
            if alias in q_tokens:
                analysis.institution_filter = full_name
                analysis.institution_alias = alias
                remaining_tokens = [t for t in remaining_tokens if t != alias]
                analysis.detected_features.append(f"institución: {alias.upper()}")
                break

        # Detectar área CONAHCyT
        for area_code, keywords in AREA_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    analysis.area_filter = area_code
                    analysis.detected_features.append(f"área: {area_code}")
                    break
            if analysis.area_filter:
                break

        # Detectar nivel SNI
        for level_code, keywords in LEVEL_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    analysis.level_filter = level_code
                    remaining_tokens = [t for t in remaining_tokens if t not in kw.split()]
                    analysis.detected_features.append(f"nivel: SNI {level_code}")
                    break
            if analysis.level_filter:
                break

        # Tokens restantes = keywords disciplinares
        analysis.discipline_keywords = [t for t in remaining_tokens if len(t) >= 3]
        analysis.clean_query = " ".join(remaining_tokens) if remaining_tokens else query

        return analysis

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """Búsqueda semántica vía FAISS."""
        query_vector = self.embedding_pipeline.encode_query(query)
        return self.faiss_store.search(query_vector, top_k=top_k)

    def _lexical_search(self, query: str, top_k: int) -> list[dict]:
        """Búsqueda léxica vía BM25."""
        return self.bm25_retriever.search_with_explanation(query, top_k=top_k)

    def _fuse_results(
        self,
        semantic: dict[str, dict],
        lexical: dict[str, dict],
    ) -> list[dict]:
        """Fusiona resultados semánticos y léxicos con score ponderado.

        Normaliza ambos scores a [0, 1] antes de combinar.
        """
        # FILTRO ESTRICTO: Solo IDs presentes en el corpus local UNAM
        all_ids = {pid for pid in (set(semantic.keys()) | set(lexical.keys()))
                   if pid in self._profiles_by_id}

        # Normalizar scores
        sem_scores = [r["score"] for pid, r in semantic.items() if pid in all_ids] if semantic else [0]
        lex_scores = [r["score"] for pid, r in lexical.items() if pid in all_ids] if lexical else [0]

        sem_max = max(sem_scores) if sem_scores and max(sem_scores) > 0 else 1.0
        lex_max = max(lex_scores) if lex_scores and max(lex_scores) > 0 else 1.0

        fused = []
        for pid in all_ids:
            sem_result = semantic.get(pid)
            lex_result = lexical.get(pid)

            sem_score = (sem_result["score"] / sem_max) if sem_result else 0.0
            lex_score = (lex_result["score"] / lex_max) if lex_result else 0.0

            final_score = (
                self.semantic_weight * sem_score +
                self.lexical_weight * lex_score
            )

            # Metadata exclusiva de la fuente local
            meta = self._profiles_by_id[pid]

            fused.append({
                "profile_id": pid,
                "score": final_score,
                "semantic_score": sem_result["score"] if sem_result else 0.0,
                "semantic_score_norm": sem_score,
                "lexical_score": lex_result["score"] if lex_result else 0.0,
                "lexical_score_norm": lex_score,
                "matched_tokens": lex_result.get("matched_tokens", []) if lex_result else [],
                "token_overlap": lex_result.get("token_overlap", 0.0) if lex_result else 0.0,
                "in_semantic": sem_result is not None,
                "in_lexical": lex_result is not None,
                **{k: v for k, v in meta.items()
                   if k not in ("score", "rank", "profile_id", "matched_tokens", "token_overlap")},
            })

        return fused

    @staticmethod
    def _merge_filters(analysis: QueryAnalysis, explicit: Optional[dict]) -> dict:
        """Combina filtros de query analysis con filtros explícitos."""
        filters = {}

        if analysis.institution_filter:
            filters["institucion"] = analysis.institution_filter
        if analysis.area_filter:
            filters["area"] = analysis.area_filter
        if analysis.level_filter:
            filters["nivel"] = analysis.level_filter

        if explicit:
            filters.update(explicit)

        return filters

    def _apply_filters(self, results: list[dict], filters: dict) -> list[dict]:
        """Aplica filtros de metadata a los resultados."""
        filtered = []

        for r in results:
            matches = True
            pid = r["profile_id"]
            # Priorizar metadata del resultado (viene del índice)
            profile = self._profiles_by_id.get(pid, r)

            for field, value in filters.items():
                if field == "institucion":
                    inst = self._normalize(profile.get("institucion", ""))
                    target = self._normalize(value)
                    if target not in inst and inst not in target:
                        # Partial match OK
                        target_tokens = set(target.split())
                        inst_tokens = set(inst.split())
                        if len(target_tokens & inst_tokens) < 2:
                            matches = False
                            break
                elif field == "area":
                    if profile.get("area", "") != value:
                        matches = False
                        break
                elif field == "nivel":
                    if profile.get("nivel", "") != value:
                        matches = False
                        break
                elif field == "disciplina":
                    disc = self._normalize(profile.get("disciplina", ""))
                    if self._normalize(value) not in disc:
                        matches = False
                        break

            if matches:
                filtered.append(r)

        return filtered

    @staticmethod
    def _apply_boosts(results: list[dict], analysis: QueryAnalysis) -> list[dict]:
        """Aplica boosts de score basados en features detectadas."""
        from src.normalizer.name_normalizer import NameNormalizer
        norm = NameNormalizer()
        
        query_norm = norm.normalize(analysis.raw_query)
        
        for r in results:
            boost = 0.0

            # 1. Lexical Boosting fuerte por coincidencia exacta de alias
            profile_aliases = r.get("aliases", [])
            for alias in profile_aliases:
                if query_norm == norm.normalize(alias):
                    boost += 0.35  # Boost fuerte para encontrar por nombre exacto
                    r["boost_reason"] = f"Match exacto de alias: {alias}"
                    break
            
            if boost == 0 and len(query_norm.split()) >= 2:
                # Intento de match parcial de nombre/apellido
                full_name_norm = r.get("normalized_name", norm.normalize(r.get("nombre_completo", "")))
                if query_norm in full_name_norm:
                    boost += 0.2
                    r["boost_reason"] = "Match parcial de nombre"

            # 2. Boost por institución match
            if analysis.institution_filter and r.get("in_lexical"):
                inst = r.get("institucion", "")
                if analysis.institution_alias and analysis.institution_alias.upper() in inst.upper():
                    boost += 0.05

            # 3. Boost por alto overlap de tokens
            if r.get("token_overlap", 0) > 0.7:
                boost += 0.03

            # 4. Boost por aparecer en ambos retrievers
            if r.get("in_semantic") and r.get("in_lexical"):
                boost += 0.05

            r["score"] = min(1.0, r["score"] + boost)
            r["boost"] = boost

        return results

    def _enrich_result(
        self,
        r: dict,
        rank: int,
        analysis: QueryAnalysis,
        include_explanation: bool,
    ) -> dict:
        """Enriquece un resultado con perfil completo + explicación."""
        profile_id = r["profile_id"]
        full_profile = self._profiles_by_id.get(profile_id, {})

        result = {
            "id": profile_id,
            "nombre_completo": r.get("nombre_completo", ""),
            "full_name": r.get("nombre_completo", ""),
            "institucion": r.get("institucion", ""),
            "institution": r.get("institucion", ""),
            "area": r.get("area", ""),
            "area_nombre": r.get("area_nombre", ""),
            "disciplina": r.get("disciplina", ""),
            "discipline": r.get("disciplina", ""),
            "nivel": r.get("nivel", ""),
            "nivel_nombre": r.get("nivel_nombre", ""),
            "dependencia": r.get("dependencia", ""),
            "nombre": full_profile.get("nombre", ""),
            "paterno": full_profile.get("paterno", ""),
            "materno": full_profile.get("materno", ""),
            "subdependencia": full_profile.get("subdependencia", ""),
            # Scores
            "score": r["score"],
            "semantic_score": r.get("semantic_score", 0.0),
            "lexical_score": r.get("lexical_score", 0.0),
            "confidence": r["score"],
            "rank": rank,
            "search_method": self._get_search_method(),
            "in_semantic": r.get("in_semantic", False),
            "in_lexical": r.get("in_lexical", False),
        }

        if include_explanation:
            result["explanation"] = self._build_explanation(r, analysis)

        return result

    def _build_explanation(self, r: dict, analysis: QueryAnalysis) -> str:
        """Construye explicación legible de por qué apareció un resultado."""
        parts = []

        # Score breakdown
        sem = r.get("semantic_score", 0)
        lex = r.get("lexical_score", 0)

        if sem > 0 and lex > 0:
            parts.append(
                f"Coincidencia híbrida: semántica={sem:.3f}, léxica={lex:.3f}"
            )
        elif sem > 0:
            parts.append(f"Coincidencia semántica: {sem:.3f}")
        elif lex > 0:
            parts.append(f"Coincidencia léxica: {lex:.3f}")

        # Matched tokens
        matched = r.get("matched_tokens", [])
        if matched:
            parts.append(f"Términos coincidentes: {', '.join(matched[:8])}")

        # Features
        if analysis.detected_features:
            parts.append(f"Filtros detectados: {', '.join(analysis.detected_features)}")

        # Boost
        boost = r.get("boost", 0)
        if boost > 0:
            reasons = []
            if r.get("boost_reason"):
                reasons.append(r["boost_reason"])
            if r.get("in_semantic") and r.get("in_lexical"):
                reasons.append("aparece en ambos métodos")
            if r.get("token_overlap", 0) > 0.7:
                reasons.append("alto overlap de tokens")
            if reasons:
                parts.append(f"Bonus: {', '.join(reasons)}")

        return " | ".join(parts) if parts else "Sin explicación disponible"

    def _get_search_method(self) -> str:
        """Determina el método de búsqueda actual."""
        if self._semantic_ready and self._lexical_ready:
            return "hybrid"
        elif self._semantic_ready:
            return "semantic"
        elif self._lexical_ready:
            return "lexical"
        return "none"

    @staticmethod
    def _normalize(text: str) -> str:
        """Normaliza texto (lowercase, sin acentos)."""
        text = text.lower().strip()
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def stats(self) -> dict:
        """Estadísticas del retriever híbrido."""
        return {
            "search_method": self._get_search_method(),
            "semantic_ready": self._semantic_ready,
            "lexical_ready": self._lexical_ready,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "total_profiles": len(self.profiles),
            "faiss_vectors": (
                self.faiss_store.index.ntotal
                if self.faiss_store and self.faiss_store.index
                else 0
            ),
            "bm25_docs": (
                len(self.bm25_retriever.profile_ids)
                if self.bm25_retriever
                else 0
            ),
        }
