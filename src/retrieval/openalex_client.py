"""
openalex_client.py -- Cliente para la API de OpenAlex.

Busca autores e instituciones usando los endpoints /authors,
/institutions y /works. Soporta polite pool via mailto,
paginacion y retry.

References:
    https://docs.openalex.org/
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from config import (
    OPENALEX_API_BASE,
    OPENALEX_EMAIL,
    OPENALEX_API_KEY,
    MAX_CANDIDATES_PER_SOURCE,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    OPENALEX_RATE_LIMIT,
)
from src.models.schemas import Candidate, CandidateSource, EvidenceVector

logger = logging.getLogger(__name__)


class OpenAlexClient:
    """Cliente para la API de OpenAlex.

    Busca autores por nombre e institucion, recupera perfiles
    completos con metricas, conceptos y publicaciones.

    Attributes:
        base_url: URL base de la API.
        session: Sesion HTTP reutilizable.
        _last_request_time: Para rate limiting.

    Examples:
        >>> client = OpenAlexClient()
        >>> candidates = client.search_authors(normalized_record)
        >>> for c in candidates:
        ...     print(f"{c.openalex_id}: {c.display_name} ({c.works_count} works)")
    """

    def __init__(self, base_url: str = OPENALEX_API_BASE) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.0 / OPENALEX_RATE_LIMIT

        # Configurar parametros globales (polite pool + API key)
        self._default_params: dict = {}
        if OPENALEX_EMAIL:
            self._default_params["mailto"] = OPENALEX_EMAIL
        if OPENALEX_API_KEY:
            self._default_params["api_key"] = OPENALEX_API_KEY

    def search_authors(
        self,
        record,
        max_results: int = MAX_CANDIDATES_PER_SOURCE,
    ) -> list[Candidate]:
        """Busca autores en OpenAlex para un investigador normalizado.

        Ejecuta multiples queries para maximizar recall:
        1. Nombre completo como search
        2. Nombre + filtro de institucion
        3. Aliases individuales

        Args:
            record: NormalizedRecord del investigador.
            max_results: Maximo de candidatos a retornar.

        Returns:
            Lista de Candidate ordenados por relevancia.
        """
        all_candidates: dict[str, Candidate] = {}

        # Q1: busqueda directa por nombre completo
        full_name = record.original.full_name
        try:
            results = self._search_authors_by_name(full_name, max_results)
            for c in results:
                if c.openalex_id and c.openalex_id not in all_candidates:
                    all_candidates[c.openalex_id] = c
        except Exception as e:
            logger.warning(f"  OpenAlex search '{full_name}' fallo: {e}")

        # Q2: busqueda con aliases (top 3)
        for alias in record.name_aliases[:3]:
            if len(all_candidates) >= max_results:
                break
            try:
                results = self._search_authors_by_name(alias, max_results=5)
                for c in results:
                    if c.openalex_id and c.openalex_id not in all_candidates:
                        all_candidates[c.openalex_id] = c
            except Exception as e:
                logger.debug(f"  OpenAlex alias '{alias}' fallo: {e}")
                continue

        candidates = list(all_candidates.values())[:max_results]
        logger.info(f"  OpenAlex: {len(candidates)} candidatos encontrados")
        return candidates

    def _search_authors_by_name(
        self, name: str, max_results: int = 10
    ) -> list[Candidate]:
        """Busca autores por nombre.

        Args:
            name: Nombre a buscar.
            max_results: Maximo de resultados.

        Returns:
            Lista de Candidate.
        """
        url = f"{self.base_url}/authors"
        params = {
            **self._default_params,
            "search": name,
            "per_page": min(max_results, 25),
        }

        data = self._request_with_retry(url, params=params)
        if not data or "results" not in data:
            return []

        return self._parse_author_results(data["results"])

    def _parse_author_results(self, results: list[dict]) -> list[Candidate]:
        """Parsea resultados de /authors a Candidate objects.

        Args:
            results: Lista de author objects de OpenAlex.

        Returns:
            Lista de Candidate.
        """
        candidates = []

        for author in results:
            openalex_id = author.get("id", "")
            display_name = author.get("display_name", "")

            if not openalex_id or not display_name:
                continue

            # Separar nombre/apellido (OpenAlex solo da display_name)
            name_parts = display_name.rsplit(" ", 1)
            given = name_parts[0] if len(name_parts) > 1 else display_name
            family = name_parts[1] if len(name_parts) > 1 else ""

            # Afiliaciones
            affiliations = []
            last_inst = author.get("last_known_institutions") or []
            if isinstance(last_inst, list):
                for inst in last_inst:
                    if isinstance(inst, dict):
                        name = inst.get("display_name", "")
                        if name:
                            affiliations.append(name)

            # Conceptos/topics
            concepts = []
            topics = author.get("topics", []) or []
            for topic in topics[:10]:
                if isinstance(topic, dict):
                    dn = topic.get("display_name", "")
                    if dn:
                        concepts.append(dn)

            # Fallback a x_concepts si topics esta vacio
            if not concepts:
                x_concepts = author.get("x_concepts", []) or []
                for xc in x_concepts[:10]:
                    if isinstance(xc, dict):
                        dn = xc.get("display_name", "")
                        if dn:
                            concepts.append(dn)

            # ORCID (OpenAlex a veces lo incluye)
            orcid_raw = author.get("orcid") or ""
            orcid_id = None
            if orcid_raw:
                # Extraer solo el ID de la URL
                orcid_id = orcid_raw.replace("https://orcid.org/", "").strip()

            candidate = Candidate(
                source=CandidateSource.OPENALEX,
                source_id=openalex_id,
                given_name=given,
                family_name=family,
                affiliations=affiliations,
                works_count=author.get("works_count", 0) or 0,
                cited_by_count=author.get("cited_by_count", 0) or 0,
                concepts=concepts,
                orcid_id=orcid_id,
                openalex_id=openalex_id,
                evidence=EvidenceVector(),
            )
            candidates.append(candidate)

        return candidates

    def get_author(self, author_id: str) -> Optional[dict]:
        """Obtiene el perfil completo de un autor.

        Args:
            author_id: OpenAlex author ID (e.g., 'A5023888391').

        Returns:
            Dict con datos completos o None.
        """
        # Normalizar ID
        if not author_id.startswith("http"):
            author_id = f"{self.base_url}/authors/{author_id}"

        url = author_id
        params = {**self._default_params}
        return self._request_with_retry(url, params=params)

    def get_works(
        self,
        author_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Obtiene publicaciones de un autor.

        Args:
            author_id: OpenAlex author ID.
            limit: Maximo de publicaciones.

        Returns:
            Lista de works como dicts.
        """
        # Extraer ID corto
        short_id = author_id
        if "openalex.org" in author_id:
            short_id = author_id.split("/")[-1]

        url = f"{self.base_url}/works"
        params = {
            **self._default_params,
            "filter": f"authorships.author.id:{short_id}",
            "sort": "cited_by_count:desc",
            "per_page": min(limit, 50),
        }

        data = self._request_with_retry(url, params=params)
        if not data or "results" not in data:
            return []

        works = []
        for w in data["results"]:
            works.append({
                "id": w.get("id", ""),
                "title": w.get("title", ""),
                "publication_year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count", 0),
                "doi": w.get("doi", ""),
                "type": w.get("type", ""),
            })

        return works

    def search_institution(self, name: str) -> Optional[dict]:
        """Busca una institucion en OpenAlex.

        Args:
            name: Nombre de la institucion.

        Returns:
            Primer resultado como dict, o None.
        """
        url = f"{self.base_url}/institutions"
        params = {
            **self._default_params,
            "search": name,
            "per_page": 1,
        }

        data = self._request_with_retry(url, params=params)
        if data and "results" in data and data["results"]:
            return data["results"][0]
        return None

    def _request_with_retry(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = MAX_RETRIES,
    ) -> Optional[dict]:
        """HTTP GET con retry y backoff exponencial.

        Args:
            url: URL del endpoint.
            params: Query parameters.
            max_retries: Numero de reintentos.

        Returns:
            JSON parseado o None.
        """
        self._rate_limit()

        for attempt in range(max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 2 ** attempt * RETRY_BACKOFF_FACTOR
                    logger.warning(f"  OpenAlex rate limited. Esperando {wait}s...")
                    time.sleep(wait)
                    continue
                elif resp.status_code == 403:
                    logger.error("  OpenAlex 403: API key requerido o invalido")
                    return None
                else:
                    logger.warning(f"  OpenAlex HTTP {resp.status_code}: {url}")
                    if attempt < max_retries:
                        time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                        continue
                    return None

            except requests.Timeout:
                logger.warning(f"  OpenAlex timeout (intento {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                continue
            except requests.RequestException as e:
                logger.error(f"  OpenAlex request error: {e}")
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
