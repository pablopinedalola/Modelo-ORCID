#!/usr/bin/env python3
"""
ingest_openalex_data.py — Descarga datos REALES de OpenAlex.

Usa el OpenAlexClient existente en src/retrieval/openalex_client.py.
Guarda JSONs individuales en data/openalex/authors/ y data/openalex/works/.
Caché incremental: si el archivo ya existe, no lo vuelve a descargar.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.openalex_client import OpenAlexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directorios de persistencia
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "openalex"
AUTHORS_DIR = DATA_DIR / "authors"
WORKS_DIR = DATA_DIR / "works"

# ---------------------------------------------------------------------------
# Lista de autores objetivo (hardcoded)
# ---------------------------------------------------------------------------
TARGET_AUTHORS = [
    "Miguel Alcubierre",
    "Carlos Gershenson",
    "José Luis Mateos",
    "Luis Mochán",
    "Alejandro Frank",
]


def safe_filename(name: str) -> str:
    """Convierte un nombre a un filename seguro."""
    return name.replace(" ", "_").replace(".", "").replace("é", "e").replace("ó", "o")


def download_author(client: OpenAlexClient, name: str) -> dict | None:
    """Busca un autor por nombre y devuelve el JSON crudo de OpenAlex."""
    url = f"{client.base_url}/authors"
    params = {**client._default_params, "search": name, "per_page": 1}
    data = client._request_with_retry(url, params=params)
    if data and data.get("results"):
        return data["results"][0]
    return None


def download_works(client: OpenAlexClient, author_oalex_id: str) -> list[dict]:
    """Descarga hasta 50 papers de un autor y extrae los campos pedidos."""
    short_id = author_oalex_id.split("/")[-1]
    url = f"{client.base_url}/works"
    params = {
        **client._default_params,
        "filter": f"authorships.author.id:{short_id}",
        "sort": "cited_by_count:desc",
        "per_page": 50,
    }
    data = client._request_with_retry(url, params=params)
    if not data or "results" not in data:
        return []

    cleaned = []
    for w in data["results"]:
        # Instituciones de todos los authorships
        institutions = set()
        for a in w.get("authorships", []):
            for inst in a.get("institutions", []):
                n = inst.get("display_name")
                if n:
                    institutions.add(n)

        # Abstract: OpenAlex devuelve abstract_inverted_index
        abstract_idx = w.get("abstract_inverted_index")
        abstract_text = None
        if abstract_idx and isinstance(abstract_idx, dict):
            # Reconstruir abstract desde inverted index
            pairs = []
            for word, positions in abstract_idx.items():
                for pos in positions:
                    pairs.append((pos, word))
            pairs.sort()
            abstract_text = " ".join(word for _, word in pairs)

        primary_loc = w.get("primary_location") or {}
        source = primary_loc.get("source") or {}

        cleaned.append({
            "openalex_id": w.get("id", ""),
            "title": w.get("title"),
            "abstract": abstract_text,
            "publication_year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
            "concepts": [
                {"id": c.get("id"), "name": c.get("display_name")}
                for c in w.get("concepts", [])
            ],
            "topics": [
                {"id": t.get("id"), "name": t.get("display_name")}
                for t in w.get("topics", [])
            ],
            "referenced_works": w.get("referenced_works", []),
            "institutions": sorted(institutions),
            "doi": w.get("doi"),
            "venue": source.get("display_name"),
        })

    return cleaned


def main():
    AUTHORS_DIR.mkdir(parents=True, exist_ok=True)
    WORKS_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAlexClient()

    logger.info("=" * 60)
    logger.info("INGEST OPENALEX — %d autores objetivo", len(TARGET_AUTHORS))
    logger.info("=" * 60)

    total_new_authors = 0
    total_new_papers = 0

    for name in TARGET_AUTHORS:
        fname = safe_filename(name)
        author_path = AUTHORS_DIR / f"{fname}.json"
        works_path = WORKS_DIR / f"{fname}_works.json"

        # ---- CACHE INCREMENTAL ----
        if author_path.exists() and works_path.exists():
            logger.info("Skipping cached author: %s", name)
            continue

        # ---- DESCARGAR AUTOR ----
        logger.info("Downloading author: %s", name)
        raw = download_author(client, name)
        if not raw:
            logger.warning("  NOT FOUND in OpenAlex: %s", name)
            continue

        oalex_id = raw.get("id", "")
        logger.info("  Found: %s  (%s)", raw.get("display_name"), oalex_id)

        with open(author_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        total_new_authors += 1

        # ---- DESCARGAR PAPERS ----
        logger.info("  Downloading papers...")
        papers = download_works(client, oalex_id)
        with open(works_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        total_new_papers += len(papers)
        logger.info("  Saved %d papers -> %s", len(papers), works_path.name)

    logger.info("=" * 60)
    logger.info(
        "DONE — new authors: %d, new papers: %d", total_new_authors, total_new_papers
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
