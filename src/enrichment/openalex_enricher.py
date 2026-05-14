import json
import logging
import os
from pathlib import Path
from typing import Optional, dict, list
from src.retrieval.openalex_client import OpenAlexClient
from config import BASE_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = BASE_DIR / "data" / "cache" / "openalex"

class OpenAlexEnricher:
    """Enriquecedor de perfiles académicos usando la API de OpenAlex con cache local."""
    
    def __init__(self):
        self.client = OpenAlexClient()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, openalex_id: str, type: str) -> Path:
        """Genera ruta de cache para un autor o sus trabajos."""
        # Limpiar ID (ej: A5043129140)
        clean_id = openalex_id.split("/")[-1]
        return CACHE_DIR / f"{clean_id}_{type}.json"

    def fetch_author_profile(self, openalex_id: str, force_refresh: bool = False) -> Optional[dict]:
        """Obtiene metadata básica del autor, con cache."""
        cache_path = self._get_cache_path(openalex_id, "profile")
        
        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error leyendo cache de perfil {openalex_id}: {e}")

        # Fetch real data
        logger.info(f"🌐 Fetching OpenAlex profile: {openalex_id}")
        url = f"{self.client.base_url}/authors/{openalex_id}"
        data = self.client._request_with_retry(url)
        
        if data:
            self._save_to_cache(cache_path, data)
            return data
        return None

    def fetch_author_works(self, openalex_id: str, force_refresh: bool = False, limit: int = 50) -> list[dict]:
        """Obtiene lista de trabajos del autor, con cache."""
        cache_path = self._get_cache_path(openalex_id, "works")
        
        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error leyendo cache de trabajos {openalex_id}: {e}")

        # Fetch real data
        logger.info(f"🌐 Fetching OpenAlex works for author: {openalex_id}")
        url = f"{self.client.base_url}/works"
        params = {
            **self.client._default_params,
            "filter": f"author.id:{openalex_id}",
            "sort": "cited_by_count:desc",
            "per_page": limit
        }
        data = self.client._request_with_retry(url, params=params)
        
        if data and "results" in data:
            results = data["results"]
            self._save_to_cache(cache_path, results)
            return results
        return []

    def _save_to_cache(self, path: Path, data: any):
        """Persiste datos en cache local."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando en cache {path}: {e}")

    def enrich_profile(self, local_profile: dict) -> dict:
        """Combina un perfil local con datos enriquecidos de OpenAlex."""
        oa_id = local_profile.get("openalex_id")
        if not oa_id:
            return local_profile

        # 1. Enriquecer metadata básica
        oa_profile = self.fetch_author_profile(oa_id)
        if oa_profile:
            local_profile["cited_by_count"] = oa_profile.get("cited_by_count", local_profile.get("cited_by_count", 0))
            local_profile["works_count"] = oa_profile.get("works_count", local_profile.get("works_count", 0))
            local_profile["orcid"] = oa_profile.get("orcid", local_profile.get("orcid", ""))
            
            # Stats (h-index, etc)
            stats = oa_profile.get("summary_stats", {})
            local_profile["summary_stats"] = stats
            
            # Topics (mezclar o reemplazar)
            oa_topics = oa_profile.get("topics", [])
            if oa_topics:
                local_profile["topics"] = oa_topics

            # Affiliations (NUEVO)
            oa_affiliations = oa_profile.get("affiliations", [])
            if oa_affiliations:
                local_profile["affiliations"] = oa_affiliations
            
            oa_lki = oa_profile.get("last_known_institutions", [])
            if oa_lki:
                local_profile["last_known_institutions"] = oa_lki

        # 2. Enriquecer trabajos
        oa_works = self.fetch_author_works(oa_id)
        if oa_works:
            # Mapear trabajos a un formato compatible con el template
            enriched_works = []
            for w in oa_works:
                enriched_works.append({
                    "id": w.get("id", "").split("/")[-1],
                    "title": w.get("title"),
                    "publication_year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "doi": w.get("doi"),
                    "venue": w.get("primary_location", {}).get("source", {}).get("display_name", ""),
                    "topics": [t.get("display_name") for t in w.get("topics", [])[:3]]
                })
            local_profile["works"] = enriched_works
            local_profile["total_works_loaded"] = len(enriched_works)

        return local_profile
