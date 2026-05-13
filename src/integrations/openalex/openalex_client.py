"""
openalex_client.py — Cliente para la API de OpenAlex con Polite Pool.

Maneja peticiones HTTP a OpenAlex con rate limiting,
retries automáticos y paginación para evitar bloqueos.
"""

from __future__ import annotations

import logging
import time
import requests
from typing import Optional, Any, Generator

from config import OPENALEX_API_BASE, OPENALEX_EMAIL, MAX_RETRIES

logger = logging.getLogger(__name__)

class OpenAlexClient:
    """Cliente para interactuar con la API de OpenAlex."""

    def __init__(self, email: str = OPENALEX_EMAIL):
        self.base_url = OPENALEX_API_BASE
        self.email = email
        self.session = requests.Session()
        self.headers = {"User-Agent": f"Modelo-ORCID/1.0 (mailto:{self.email})"}

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Realiza una petición GET con retries y rate limiting básico."""
        url = f"{self.base_url}/{endpoint}"
        p = params or {}
        if self.email:
            p["mailto"] = self.email

        for attempt in range(MAX_RETRIES):
            try:
                # Polite pool: 10 req/seg = 0.1s entre llamadas. Damos 0.15s por seguridad.
                time.sleep(0.15)
                res = self.session.get(url, params=p, headers=self.headers, timeout=15)
                res.raise_for_status()
                return res.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error OpenAlex API ({url}): {e}. Retrying {attempt+1}/{MAX_RETRIES}...")
                time.sleep(2 ** attempt)

        logger.error(f"Fallo en OpenAlex API tras {MAX_RETRIES} intentos: {url}")
        return {}

    def get_author(self, orcid_or_id: str) -> dict:
        """Obtiene un autor por ORCID (https://orcid.org/...) o OpenAlex ID."""
        if orcid_or_id.startswith("https://orcid.org/"):
            endpoint = f"authors/{orcid_or_id}"
        else:
            endpoint = f"authors/{orcid_or_id}"
        return self._get(endpoint)

    def search_authors(self, query: str, limit: int = 10) -> list[dict]:
        """Busca autores por nombre."""
        data = self._get("authors", {"search": query, "per-page": limit})
        return data.get("results", [])

    def get_works(self, author_id: str, limit: int = 50) -> list[dict]:
        """Obtiene trabajos de un autor."""
        # author_id format is usually the full URL or the 'A...' id
        author_id_short = author_id.split("/")[-1] if "/" in author_id else author_id
        
        # Iterar sobre las paginas
        results = []
        page = 1
        while True:
            data = self._get("works", {
                "filter": f"author.id:{author_id_short}",
                "per-page": min(limit - len(results), 50),
                "page": page
            })
            
            items = data.get("results", [])
            if not items:
                break
                
            results.extend(items)
            
            if len(results) >= limit or not data.get("meta", {}).get("next_cursor"):
                if len(results) >= limit or page * 50 >= data.get("meta", {}).get("count", 0):
                    break
            page += 1
            
        return results[:limit]

    def get_institution(self, ror_id: str) -> dict:
        """Obtiene institucion por ROR."""
        if ror_id.startswith("https://ror.org/"):
            endpoint = f"institutions/{ror_id}"
        else:
            endpoint = f"institutions/https://ror.org/{ror_id}"
        return self._get(endpoint)
