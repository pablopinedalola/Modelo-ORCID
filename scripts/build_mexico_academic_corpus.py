#!/usr/bin/env python3
"""
build_mexico_academic_corpus.py — Expande el corpus con datos reales de OpenAlex.

Descarga autores e instituciones de México (UNAM, IPN, UAM, etc.) y los
persiste para su indexación en el pipeline RAG.
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
logger = logging.getLogger("build_mexico_corpus")

# IDs de instituciones en OpenAlex
INSTITUTIONS = {
    "UNAM": "I8961855",
    "IPN": "I59361560",
    "UAM": "I200362191",
    "CINVESTAV": "I68368234",
    "TEC_MTY": "I98461037"
}

def normalize_text(text: str) -> str:
    """Normaliza texto para searchable_text."""
    if not text: return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    return text

# Investigadores clave que deben estar en el corpus
MUST_HAVE_AUTHORS = [
    "Miguel Alcubierre",
    "Humberto Carrillo Calvet",
    "José Luis Andrade",
    "Julieta Fierro",
    "Arturo Menchaca",
    "Deborah Dultzin",
]

class MexicoCorpusBuilder:
    def __init__(self):
        self.client = OpenAlexClient()
        from src.normalizer.name_normalizer import NameNormalizer
        self.normalizer = NameNormalizer()
        self.authors_file = PROCESSED_DATA_DIR / "mexico_authors.json"
        self.works_file = PROCESSED_DATA_DIR / "mexico_works.json"
        self.authors = []
        self.works = []
        self.seen_author_ids = set()

    def fetch_must_have_authors(self):
        """Descarga investigadores específicos por nombre."""
        for name in MUST_HAVE_AUTHORS:
            logger.info(f"Buscando investigador Must-Have: {name}...")
            url = f"{self.client.base_url}/authors"
            params = {
                **self.client._default_params,
                "filter": f"display_name.search:{name}",
                "sort": "cited_by_count:desc"
            }
            data = self.client._request_with_retry(url, params=params)
            if data and "results" in data and data["results"]:
                # Tomar el primero (más citado)
                self._process_authors([data["results"][0]], "Must-Have")

    def fetch_mexican_authors(self, limit_per_inst: int = 100):
        """Descarga autores de las instituciones principales."""
        for name, inst_id in INSTITUTIONS.items():
            logger.info(f"Descargando autores de {name} ({inst_id})...")
            
            url = f"{self.client.base_url}/authors"
            params = {
                **self.client._default_params,
                "filter": f"last_known_institutions.id:{inst_id}",
                "sort": "cited_by_count:desc",
                "per_page": min(limit_per_inst, 50)
            }
            
            downloaded = 0
            while downloaded < limit_per_inst:
                data = self.client._request_with_retry(url, params=params)
                if not data or "results" not in data:
                    break
                
                results = data["results"]
                if not results:
                    break
                
                self._process_authors(results, name)
                downloaded += len(results)
                if len(results) < params["per_page"]:
                    break
                break 

    def _process_authors(self, results, inst_label):
        """Procesa y mapea resultados de autores."""
        for author in results:
            aid = author.get("id", "").split("/")[-1]
            if aid in self.seen_author_ids:
                continue
            self.seen_author_ids.add(aid)

            display_name = author.get("display_name", "")
            inst_info = (author.get("last_known_institutions") or [{}])
            inst_name = inst_info[0].get("display_name", inst_label) if inst_info else inst_label
            
            # Generar aliases y nombre normalizado
            # Intentar separar nombre/apellidos del display_name
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
            norm_name = self.normalizer.normalize(display_name)
            
            topics = [t.get("display_name", "") for t in author.get("topics", [])[:3]]
            disciplina = topics[0] if topics else ""
            
            search_text = f"{display_name} {' '.join(aliases)} {inst_name} {disciplina} {' '.join(topics)} OpenAlex"
            
            author_mapped = {
                "id": f"oa_{aid}",
                "nombre_completo": display_name,
                "normalized_name": norm_name,
                "aliases": aliases,
                "institucion": inst_name,
                "dependencia": "OpenAlex Enrichment",
                "area_nombre": "Enriquecimiento Externo",
                "disciplina": disciplina,
                "nivel_nombre": f"Citas: {author.get('cited_by_count', 0)}",
                "searchable_text": normalize_text(search_text),
                "works_count": author.get("works_count", 0),
                "cited_by_count": author.get("cited_by_count", 0),
                "h_index": author.get("summary_stats", {}).get("h_index", 0),
                "orcid": author.get("orcid", ""),
                "topics": author.get("topics", [])[:5],
                "affiliations": author.get("affiliations", [])
            }
            self.authors.append(author_mapped)
            self.fetch_author_works(aid)
    def fetch_author_works(self, author_id: str):
        """Descarga los trabajos mas relevantes de un autor."""
        logger.info(f"  Descargando works para {author_id}...")
        url = f"{self.client.base_url}/works"
        params = {
            **self.client._default_params,
            "filter": f"authorships.author.id:{author_id}",
            "sort": "cited_by_count:desc",
            "per_page": 10
        }
        data = self.client._request_with_retry(url, params=params)
        if data and "results" in data:
            for w in data["results"]:
                # Limpiar abstract
                abstract_idx = w.get("abstract_inverted_index")
                abstract_text = ""
                if abstract_idx:
                    try:
                        pairs = []
                        for word, positions in abstract_idx.items():
                            for pos in positions:
                                pairs.append((pos, word))
                        pairs.sort()
                        abstract_text = " ".join(word for _, word in pairs)
                    except: pass

                work_mapped = {
                    "openalex_id": w.get("id"),
                    "title": w.get("title"),
                    "abstract": abstract_text,
                    "publication_year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "doi": w.get("doi"),
                    "topics": [t.get("display_name") for t in w.get("topics", [])[:5]],
                    "institutions": [inst.get("display_name") for a in w.get("authorships", []) for inst in a.get("institutions", [])],
                    "author_id": f"oa_{author_id}"
                }
                self.works.append(work_mapped)

    def save(self):
        """Guarda los resultados en disco."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(self.authors_file, "w", encoding="utf-8") as f:
            json.dump(self.authors, f, indent=2, ensure_ascii=False)
        
        with open(self.works_file, "w", encoding="utf-8") as f:
            json.dump(self.works, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Proceso finalizado. Autores: {len(self.authors)}, Works: {len(self.works)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Autores por institucion")
    args = parser.parse_args()

    builder = MexicoCorpusBuilder()
    builder.fetch_must_have_authors()
    builder.fetch_mexican_authors(limit_per_inst=args.limit)
    builder.save()
    
    # 7. Regenerar índices automáticamente.
    logger.info("Regenerando índices vectoriales...")
    python_exec = "./venv/bin/python3" if os.path.exists("./venv/bin/python3") else "python3"
    os.system(f"{python_exec} scripts/build_vector_index.py")

if __name__ == "__main__":
    main()
