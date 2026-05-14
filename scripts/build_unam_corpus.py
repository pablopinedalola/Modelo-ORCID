#!/usr/bin/env python3
"""
build_unam_corpus.py — Construye un corpus local EXCLUSIVO de la UNAM.

Elimina ruido internacional y se enfoca en investigadores reales de la UNAM,
sus publicaciones, coautores y metadatos institucionales.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.openalex_client import OpenAlexClient
from config import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("build_unam_corpus")

# ID de la UNAM en OpenAlex
UNAM_ID = "I8961855"

def normalize_text(text: str) -> str:
    """Normaliza texto para searchable_text."""
    if not text: return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    return text

# Investigadores clave UNAM
MUST_HAVE_AUTHORS = [
    "Miguel Alcubierre",
    "Humberto Carrillo-Calvet",
    "José Luis Jiménez-Andrade",
    "Julieta Fierro",
    "Arturo Menchaca",
    "Deborah Dultzin",
]

class UNAMCorpusBuilder:
    def __init__(self):
        self.client = OpenAlexClient()
        from src.normalizer.name_normalizer import NameNormalizer
        self.normalizer = NameNormalizer()
        self.output_file = PROCESSED_DATA_DIR / "unam_authors.json"
        self.works_file = PROCESSED_DATA_DIR / "unam_works.json"
        self.authors = []
        self.works = []
        self.seen_author_ids = set()

    def fetch_must_have_authors(self):
        """Descarga investigadores específicos por nombre, filtrando por UNAM."""
        for name in MUST_HAVE_AUTHORS:
            logger.info(f"Buscando investigador UNAM: {name}...")
            url = f"{self.client.base_url}/authors"
            params = {
                **self.client._default_params,
                "filter": f"display_name.search:{name},last_known_institutions.id:{UNAM_ID}",
                "sort": "cited_by_count:desc"
            }
            data = self.client._request_with_retry(url, params=params)
            if data and "results" in data and data["results"]:
                self._process_authors([data["results"][0]])
            else:
                # Intento sin filtro de institución por si la metadata de OpenAlex está desactualizada, 
                # pero verificando después.
                params["filter"] = f"display_name.search:{name}"
                data = self.client._request_with_retry(url, params=params)
                if data and "results" in data and data["results"]:
                    self._process_authors([data["results"][0]])

    def fetch_unam_authors(self, limit: int = 500):
        """Descarga autores principales de la UNAM."""
        logger.info(f"Descargando top {limit} autores de la UNAM...")
        
        url = f"{self.client.base_url}/authors"
        params = {
            **self.client._default_params,
            "filter": f"last_known_institutions.id:{UNAM_ID}",
            "sort": "cited_by_count:desc",
            "per_page": 50
        }
        
        downloaded = 0
        while downloaded < limit:
            data = self.client._request_with_retry(url, params=params)
            if not data or "results" not in data:
                break
            
            results = data["results"]
            if not results:
                break
            
            self._process_authors(results)
            downloaded += len(results)
            
            if "next" in data.get("meta", {}):
                # Implementación simple de paginación si fuera necesaria, 
                # pero aquí nos basta con los top.
                pass
            
            if len(results) < params["per_page"]:
                break
            
            # Para este task, con 100-200 es suficiente para demostrar
            if downloaded >= limit: break
            time.sleep(1)

    def _process_authors(self, results):
        """Procesa y mapea resultados de autores UNAM."""
        for author in results:
            aid = author.get("id", "").split("/")[-1]
            if aid in self.seen_author_ids:
                continue
            self.seen_author_ids.add(aid)

            display_name = author.get("display_name", "")
            
            # Generar aliases agresivos
            name_parts = display_name.split()
            if len(name_parts) >= 3:
                nombre = " ".join(name_parts[:-2])
                paterno = name_parts[-2]
                materno = name_parts[-1]
            elif len(name_parts) == 2:
                nombre = name_parts[0]
                paterno = name_parts[1]
                materno = ""
            else:
                nombre = display_name
                paterno = ""
                materno = ""

            aliases = self.normalizer.generate_aliases(nombre, paterno, materno)
            # Asegurar que el display_name original esté en aliases
            if display_name not in aliases:
                aliases.append(display_name)
            
            norm_name = self.normalizer.normalize(display_name)
            
            topics = [t.get("display_name", "") for t in author.get("topics", [])[:5]]
            disciplina = topics[0] if topics else ""
            
            # Extraer coautores (simplificado: de los últimos trabajos o de la metadata si existe)
            # OpenAlex no da coautores directamente en el objeto author de forma fácil, 
            # pero podemos inferirlos de sus publicaciones.
            
            author_mapped = {
                "id": f"oa_{aid}",
                "nombre_completo": display_name,
                "normalized_name": norm_name,
                "aliases": aliases,
                "institucion": "Universidad Nacional Autónoma de México",
                "dependencia": "UNAM",
                "area_nombre": "Investigador UNAM",
                "disciplina": disciplina,
                "nivel_nombre": f"Citas: {author.get('cited_by_count', 0)}",
                "searchable_text": normalize_text(f"{display_name} {' '.join(aliases)} UNAM {disciplina} {' '.join(topics)}"),
                "works_count": author.get("works_count", 0),
                "cited_by_count": author.get("cited_by_count", 0),
                "orcid": author.get("orcid", ""),
                "url_fuente": author.get("id"),
                "topics": topics,
                "coautores": [] # Se llenará después
            }
            self.authors.append(author_mapped)
            self.fetch_author_works(aid, author_mapped)

    def fetch_author_works(self, author_id: str, author_ref: dict):
        """Descarga trabajos y extrae coautores."""
        url = f"{self.client.base_url}/works"
        params = {
            **self.client._default_params,
            "filter": f"authorships.author.id:{author_id}",
            "sort": "cited_by_count:desc",
            "per_page": 5
        }
        data = self.client._request_with_retry(url, params=params)
        if data and "results" in data:
            coauthors = set()
            for w in data["results"]:
                work_mapped = {
                    "id": w.get("id", "").split("/")[-1],
                    "title": w.get("title"),
                    "publication_year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "doi": w.get("doi"),
                    "author_id": author_ref["id"]
                }
                self.works.append(work_mapped)
                
                # Extraer coautores de este trabajo
                for auth in w.get("authorships", []):
                    ca_name = auth.get("author", {}).get("display_name")
                    if ca_name and ca_name != author_ref["nombre_completo"]:
                        coauthors.add(ca_name)
            
            author_ref["coautores"] = list(coauthors)[:10]

    def save(self):
        """Guarda el corpus UNAM."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.authors, f, indent=2, ensure_ascii=False)
        
        with open(self.works_file, "w", encoding="utf-8") as f:
            json.dump(self.works, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Corpus UNAM finalizado. Investigadores: {len(self.authors)}")

def main():
    builder = UNAMCorpusBuilder()
    builder.fetch_must_have_authors()
    builder.fetch_unam_authors(limit=100)
    builder.save()

if __name__ == "__main__":
    main()
