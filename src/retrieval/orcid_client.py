"""
orcid_client.py -- Cliente para la ORCID Public API v3.0.

Busca investigadores por nombre y afiliacion usando el endpoint
expanded-search con sintaxis SOLR. Soporta retry con backoff
exponencial y rate limiting basico.

La ORCID Public API no requiere token para busquedas basicas,
pero si para obtener registros completos.

References:
    https://info.orcid.org/documentation/api-tutorials/api-tutorial-searching-the-orcid-registry/
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from config import (
    ORCID_API_BASE,
    MAX_CANDIDATES_PER_SOURCE,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    ORCID_RATE_LIMIT,
)
from src.models.schemas import Candidate, CandidateSource, EvidenceVector

logger = logging.getLogger(__name__)


class ORCIDClient:
    """Cliente para la ORCID Public API v3.0.

    Busca candidatos usando expanded-search con queries SOLR
    combinando nombre, apellido y afiliacion.

    Attributes:
        base_url: URL base de la API.
        session: Sesion HTTP reutilizable.
        _last_request_time: Timestamp del ultimo request (rate limiting).

    Examples:
        >>> client = ORCIDClient()
        >>> candidates = client.search_researcher(normalized_record)
        >>> for c in candidates:
        ...     print(f"{c.orcid_id}: {c.display_name}")
    """

    HEADERS = {
        "Accept": "application/json",
    }

    def __init__(self, base_url: str = ORCID_API_BASE) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.0 / ORCID_RATE_LIMIT

    def search_researcher(
        self,
        record,
        max_results: int = MAX_CANDIDATES_PER_SOURCE,
    ) -> list[Candidate]:
        """Busca candidatos ORCID para un investigador normalizado.

        Ejecuta multiples queries (nombre completo, con afiliacion,
        solo apellido) para maximizar recall.

        Args:
            record: NormalizedRecord del investigador.
            max_results: Maximo de candidatos a retornar.

        Returns:
            Lista de Candidate ordenados por relevancia ORCID.
        """
        all_candidates: dict[str, Candidate] = {}

        # Query 1: nombre + apellido + afiliacion
        queries = self._build_queries(record)

        for i, query in enumerate(queries):
            try:
                results = self._execute_search(query, max_results=max_results)
                for c in results:
                    if c.orcid_id not in all_candidates:
                        all_candidates[c.orcid_id] = c
            except Exception as e:
                logger.warning(f"  ORCID query {i+1} fallo: {e}")
                continue

            if len(all_candidates) >= max_results:
                break

        candidates = list(all_candidates.values())[:max_results]
        logger.info(f"  ORCID: {len(candidates)} candidatos encontrados")
        return candidates

    def _build_queries(self, record) -> list[str]:
        """Construye queries SOLR para la busqueda.

        Genera multiples queries de mayor a menor especificidad:
        1. Apellido + nombre + institucion
        2. Apellido + nombre
        3. Apellido + inicial + institucion
        4. Solo apellido + institucion

        Args:
            record: NormalizedRecord.

        Returns:
            Lista de query strings SOLR.
        """
        original = record.original
        paterno = original.paterno.strip()
        nombre = original.nombre.strip().split()[0] if original.nombre else ""

        queries = []

        # Buscar aliases de institucion mas cortos para query
        inst_query = ""
        if record.institution_aliases:
            # Usar el alias mas corto (usualmente la abreviatura)
            short = min(record.institution_aliases, key=len)
            if len(short) <= 30:
                inst_query = short

        # Q1: apellido + nombre + institucion
        if paterno and nombre and inst_query:
            q = (
                f'family-name:"{paterno}" AND '
                f'given-names:"{nombre}" AND '
                f'affiliation-org-name:"{inst_query}"'
            )
            queries.append(q)

        # Q2: apellido + nombre (sin institucion)
        if paterno and nombre:
            q = f'family-name:"{paterno}" AND given-names:"{nombre}"'
            queries.append(q)

        # Q3: apellido + inicial + institucion
        if paterno and nombre and inst_query:
            initial = nombre[0]
            q = (
                f'family-name:"{paterno}" AND '
                f'given-names:"{initial}*" AND '
                f'affiliation-org-name:"{inst_query}"'
            )
            queries.append(q)

        # Q4: solo apellido + institucion
        if paterno and inst_query:
            q = (
                f'family-name:"{paterno}" AND '
                f'affiliation-org-name:"{inst_query}"'
            )
            queries.append(q)

        return queries

    def _execute_search(
        self, query: str, max_results: int = 10
    ) -> list[Candidate]:
        """Ejecuta una query contra expanded-search.

        Args:
            query: Query string SOLR.
            max_results: Numero maximo de resultados.

        Returns:
            Lista de Candidate parseados de la respuesta.
        """
        url = f"{self.base_url}/expanded-search/"
        params = {
            "q": query,
            "rows": min(max_results, 50),
            "start": 0,
        }

        logger.debug(f"  ORCID query: {query}")
        data = self._request_with_retry(url, params=params)

        if not data:
            return []

        results = data.get("expanded-result")
        if not results or not isinstance(results, list):
            return []

        return self._parse_candidates(results)

    def _parse_candidates(self, results: list[dict]) -> list[Candidate]:
        """Parsea la respuesta expanded-search a objetos Candidate.

        Args:
            results: Lista de resultados de expanded-search.

        Returns:
            Lista de Candidate.
        """
        candidates = []

        for r in results:
            orcid_id = r.get("orcid-id", "")
            if not orcid_id:
                continue

            given = r.get("given-names", "") or ""
            family = r.get("family-names", "") or ""

            # Extraer afiliaciones
            affiliations = []
            inst_names = r.get("institution-name", [])
            if inst_names:
                if isinstance(inst_names, list):
                    affiliations = [str(n) for n in inst_names if n]
                elif isinstance(inst_names, str):
                    affiliations = [inst_names]

            candidate = Candidate(
                source=CandidateSource.ORCID,
                source_id=orcid_id,
                given_name=given,
                family_name=family,
                affiliations=affiliations,
                orcid_id=orcid_id,
                evidence=EvidenceVector(),
            )
            candidates.append(candidate)

        return candidates

    def get_record(self, orcid_id: str) -> Optional[dict]:
        """Obtiene el registro completo de un ORCID iD.

        Args:
            orcid_id: ORCID iD (e.g., '0000-0002-1234-5678').

        Returns:
            Dict con datos del registro o None si falla.
        """
        url = f"{self.base_url}/{orcid_id}/record"
        try:
            data = self._request_with_retry(url)
            return data
        except Exception as e:
            logger.warning(f"  No se pudo obtener registro ORCID {orcid_id}: {e}")
            return None

    def _request_with_retry(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = MAX_RETRIES,
    ) -> Optional[dict]:
        """Ejecuta un HTTP GET con retry y backoff exponencial.

        Args:
            url: URL del endpoint.
            params: Query parameters.
            max_retries: Numero maximo de reintentos.

        Returns:
            Respuesta JSON parseada, o None.

        Raises:
            requests.RequestException: Si todos los reintentos fallan.
        """
        # Rate limiting
        self._rate_limit()

        for attempt in range(max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    logger.debug(f"  ORCID 404: {url}")
                    return None
                elif resp.status_code == 429:
                    wait = 2 ** attempt * RETRY_BACKOFF_FACTOR
                    logger.warning(f"  ORCID rate limited. Esperando {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(
                        f"  ORCID HTTP {resp.status_code}: {url}"
                    )
                    if attempt < max_retries:
                        time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                        continue
                    return None

            except requests.Timeout:
                logger.warning(f"  ORCID timeout (intento {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                continue
            except requests.RequestException as e:
                logger.error(f"  ORCID request error: {e}")
                if attempt < max_retries:
                    time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                    continue
                raise

        return None

    def _rate_limit(self) -> None:
        """Aplica rate limiting basico."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
