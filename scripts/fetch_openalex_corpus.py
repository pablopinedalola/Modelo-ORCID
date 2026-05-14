#!/usr/bin/env python3
"""
fetch_openalex_corpus.py -- Script para descargar papers reales de OpenAlex.
Descarga datos para el sistema RAG sobre áreas de ciencia e inteligencia artificial en la UNAM.
"""

import json
import logging
import requests
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("fetch_openalex")

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WORKS_OUT = PROCESSED_DIR / "openalex_works.json"
AUTHORS_OUT = PROCESSED_DIR / "openalex_authors.json"

# Constantes
API_BASE = "https://api.openalex.org"
EMAIL = "modelo-orcid@example.com"
MAX_WORKS = 5000
MAX_AUTHORS = 1000

def fetch_data():
    logger.info("Iniciando descarga de OpenAlex...")
    
    # UNAM ID
    unam_id = "I74973139"
    search_terms = [
        "machine learning", "deep learning", "graph neural networks",
        "artificial intelligence", "computer vision", "NLP", "transformers",
        "mathematics", "physics", "quantum computing", "network science"
    ]
    
    works_list = []
    authors_dict = {}
    
    # Set of IDs to avoid duplicates
    seen_works = set()
    
    for term in search_terms:
        if len(works_list) >= MAX_WORKS:
            break
            
        logger.info(f"Buscando works para: {term}")
        
        url = f"{API_BASE}/works"
        params = {
            "filter": f"institutions.id:{unam_id},default.search:{term}",
            "per-page": 25,
            "mailto": EMAIL
        }
        
        page = 1
        while len(works_list) < MAX_WORKS and page <= 3:
            params["page"] = page
            logger.info(f"  Página {page} ({term})...")
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"  Error en consulta: {e}")
                break
                
            items = data.get("results", [])
            if not items:
                logger.info(f"  Sin más resultados para {term}.")
                break
                
            logger.info(f"  {len(items)} resultados obtenidos.")
            
            for item in items:
                if len(works_list) >= MAX_WORKS:
                    break
                    
                work_id = item.get("id", "")
                title = item.get("title", "")
                if not title or work_id in seen_works:
                    continue
                    
                seen_works.add(work_id)
                
                abstract_idx = item.get("abstract_inverted_index", {})
                abs_text = ""
                if abstract_idx:
                    words = {}
                    for word, positions in abstract_idx.items():
                        for pos in positions:
                            words[pos] = word
                    abs_text = " ".join([words[i] for i in sorted(words.keys())])
                    
                work_authors = []
                work_insts = set()
                for auth in item.get("authorships", []):
                    author_name = auth.get("author", {}).get("display_name", "")
                    author_id = auth.get("author", {}).get("id", "")
                    if author_name:
                        work_authors.append(author_name)
                    
                    if author_id and len(authors_dict) < MAX_AUTHORS:
                        if author_id not in authors_dict:
                            authors_dict[author_id] = {
                                "id": author_id,
                                "openalex_id": author_id,
                                "name": author_name,
                                "institutions": [],
                                "works_count": 1
                            }
                        else:
                            authors_dict[author_id]["works_count"] += 1
                    
                    for inst in auth.get("institutions", []):
                        inst_name = inst.get("display_name")
                        if inst_name:
                            work_insts.add(inst_name)
                            if author_id in authors_dict and inst_name not in authors_dict[author_id]["institutions"]:
                                authors_dict[author_id]["institutions"].append(inst_name)
                                
                topics = [c.get("display_name") for c in item.get("concepts", []) if c.get("score", 0) > 0.4]
                
                source = item.get("primary_location", {}).get("source", {}) if item.get("primary_location") else {}
                journal = source.get("display_name", "") if source else ""
                
                # Generar searchable_text
                searchable_text = f"{title}. {abs_text}. Conceptos: {', '.join(topics)}. Autores: {', '.join(work_authors)}. Instituciones: {', '.join(work_insts)}."
                
                works_list.append({
                    "id": work_id,
                    "openalex_id": work_id,
                    "title": title,
                    "abstract": abs_text,
                    "authors": work_authors,
                    "institutions": list(work_insts),
                    "cited_by_count": item.get("cited_by_count", 0),
                    "publication_year": item.get("publication_year"),
                    "concepts": topics,
                    "source": journal,
                    "doi": item.get("doi"),
                    "searchable_text": searchable_text
                })
                
            page += 1
        
    logger.info(f"Guardando {len(works_list)} works y {len(authors_dict)} authors.")
    
    with open(WORKS_OUT, "w", encoding="utf-8") as f:
        json.dump(works_list, f, ensure_ascii=False, indent=2)
        
    with open(AUTHORS_OUT, "w", encoding="utf-8") as f:
        json.dump(list(authors_dict.values()), f, ensure_ascii=False, indent=2)
        
    logger.info("Proceso completado exitosamente.")

if __name__ == "__main__":
    fetch_data()
