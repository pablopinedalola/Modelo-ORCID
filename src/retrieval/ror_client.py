"""
ror_client.py -- Cliente para la API de ROR (Research Organization Registry).

Resuelve nombres de instituciones mexicanas a ROR IDs usando
el endpoint de busqueda y el endpoint de affiliation matching.
No requiere autenticacion.

References:
    https://ror.readme.io/v2/docs/api-v2
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from config import (
    ROR_API_BASE,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    ROR_RATE_LIMIT,
)

logger = logging.getLogger(__name__)


class RORClient:
    """Cliente para la ROR API v2.

    Busca instituciones por nombre y resuelve a ROR IDs.
    Mantiene un cache local para evitar queries repetidas.

    Attributes:
        base_url: URL base de la API.
        session: Sesion HTTP.
        _cache: Cache de resultados {query: result}.

    Examples:
        >>> client = RORClient()
        >>> result = client.search_institution("UNAM")
        >>> print(result["id"], result["name"])
    """

    def __init__(self, base_url: str = ROR_API_BASE) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._cache: dict[str, Optional[dict]] = {}
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.0 / ROR_RATE_LIMIT

    def search_institution(self, name: str) -> Optional[dict]:
        """Busca una institucion por nombre.

        Args:
            name: Nombre de la institucion (puede ser abreviatura).

        Returns:
            Dict con id, name, country, types, links del primer
            resultado relevante, o None si no hay match.

        Examples:
            >>> client = RORClient()
            >>> r = client.search_institution("Universidad Nacional Autonoma de Mexico")
            >>> r["id"]
            'https://ror.org/01tmp8f25'
        """
        cache_key = name.strip().lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._query_search(name)
        self._cache[cache_key] = result
        return result

    def _query_search(self, query: str) -> Optional[dict]:
        """Ejecuta query contra el endpoint de busqueda.

        Args:
            query: Texto de busqueda.

        Returns:
            Dict normalizado del primer resultado o None.
        """
        url = self.base_url
        params = {"query": query}

        data = self._request_with_retry(url, params=params)
        if not data:
            return None

        items = data.get("items", [])
        if not items:
            logger.debug(f"  ROR: sin resultados para '{query}'")
            return None

        # Tomar el primer resultado (mayor relevancia)
        org = items[0].get("organization", items[0])
        return self._normalize_result(org)

    def affiliation_match(self, institution_name: str) -> Optional[dict]:
        """Usa el endpoint de affiliation para matching fuzzy.

        Este endpoint esta disenado para resolver nombres de
        afiliaciones de publicaciones a organizaciones ROR.

        Args:
            institution_name: Nombre de la afiliacion.

        Returns:
            Dict normalizado o None.
        """
        cache_key = f"aff:{institution_name.strip().lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = self.base_url
        params = {"affiliation": institution_name}

        data = self._request_with_retry(url, params=params)
        if not data:
            self._cache[cache_key] = None
            return None

        items = data.get("items", [])
        if not items:
            self._cache[cache_key] = None
            return None

        # Buscar el resultado con mayor score
        best = None
        best_score = 0.0
        for item in items:
            score = item.get("score", 0)
            if score > best_score:
                best_score = score
                best = item

        if best and best_score > 0.5:
            org = best.get("organization", best)
            result = self._normalize_result(org)
            result["match_score"] = best_score
            self._cache[cache_key] = result
            return result

        self._cache[cache_key] = None
        return None

    def get_organization(self, ror_id: str) -> Optional[dict]:
        """Obtiene detalles de una organizacion por ROR ID.

        Args:
            ror_id: ROR ID (URL o ID corto).

        Returns:
            Dict con datos completos de la organizacion.
        """
        # Normalizar: extraer ID corto si es URL
        if "ror.org/" in ror_id:
            short_id = ror_id.split("ror.org/")[-1]
        else:
            short_id = ror_id

        url = f"{self.base_url}/{short_id}"
        data = self._request_with_retry(url)
        if not data:
            return None

        return self._normalize_result(data)

    @staticmethod
    def _normalize_result(org: dict) -> dict:
        """Normaliza un resultado ROR a un dict limpio.

        Args:
            org: Organization object de la API ROR.

        Returns:
            Dict con campos estandarizados.
        """
        # Extraer nombre (v2 usa 'names' array)
        name = ""
        names = org.get("names", [])
        if names:
            # Preferir el nombre "ror_display"
            for n in names:
                if isinstance(n, dict):
                    types = n.get("types", [])
                    if "ror_display" in types:
                        name = n.get("value", "")
                        break
            if not name and names:
                first = names[0]
                name = first.get("value", "") if isinstance(first, dict) else str(first)
        if not name:
            name = org.get("name", "")

        # Extraer pais
        country = ""
        locations = org.get("locations", [])
        if locations and isinstance(locations[0], dict):
            geo = locations[0].get("geonames_details", {})
            country = geo.get("country_name", "")
        if not country:
            country_info = org.get("country", {})
            country = country_info.get("country_name", "") if isinstance(country_info, dict) else ""

        # Extraer links
        links = org.get("links", [])
        if isinstance(links, list):
            links = [l.get("value", l) if isinstance(l, dict) else str(l) for l in links]

        return {
            "id": org.get("id", ""),
            "name": name,
            "country": country,
            "types": org.get("types", []),
            "links": links,
            "established": org.get("established"),
        }

    def _request_with_retry(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = MAX_RETRIES,
    ) -> Optional[dict]:
        """HTTP GET con retry."""
        self._rate_limit()

        for attempt in range(max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=10)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 2 ** attempt * RETRY_BACKOFF_FACTOR
                    logger.warning(f"  ROR rate limited. Esperando {wait}s...")
                    time.sleep(wait)
                    continue
                elif resp.status_code == 404:
                    return None
                else:
                    logger.warning(f"  ROR HTTP {resp.status_code}: {url}")
                    if attempt < max_retries:
                        time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                        continue
                    return None

            except requests.Timeout:
                logger.warning(f"  ROR timeout (intento {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                continue
            except requests.RequestException as e:
                logger.error(f"  ROR request error: {e}")
                if attempt < max_retries:
                    time.sleep(RETRY_BACKOFF_FACTOR * (attempt + 1))
                    continue
                raise

        return None

    def _rate_limit(self) -> None:
        """Aplica rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
