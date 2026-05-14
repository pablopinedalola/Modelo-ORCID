import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

from src.normalizer.name_normalizer import NameNormalizer
from config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

class UNAMRepositoryScraper:
    """Scraper para el Repositorio Institucional de la UNAM y revistas relacionadas."""
    
    def __init__(self):
        self.base_url = "https://repositorio.unam.mx"
        self.normalizer = NameNormalizer()
        self.authors = {} # name_norm -> author_data
        self.works = []
        self.processed_count = 0 # Contador de documentos procesados correctamente
        
    def scrape_by_area(self, area_id: str, limit_pages: int = 1):
        """Scrapea investigadores y trabajos por área temática. Limitado a 1 página y 5 resultados."""
        search_url = f"{self.base_url}/contenidos"
        
        # PROCESAR MÁXIMO: 1 página
        for page in range(1, 2):
            logger.info(f"Scrapeando página {page} del repositorio UNAM...")
            
            params = {
                "q": area_id,
                "i": page,
                "v": "0",
                "t": "search_0",
                "as": "0"
            }
            
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                }
                
                response = requests.get(search_url, params=params, headers=headers, timeout=20)
                response.encoding = "utf-8"
                
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Selectores para encontrar los resultados
                records = soup.select(".cont-record-min") or soup.select(".gral-container-record-min") or soup.select("div[class*='record']")
                
                if not records:
                    break
                    
                # PROCESAR MÁXIMO: 5 resultados
                processed_in_page = 0
                for record in records:
                    if processed_in_page >= 5:
                        break
                        
                    title_elem = record.select_one("a.cont-text-title-record-min") or record.select_one("a")
                    if not title_elem: continue
                    
                    relative_url = title_elem.get('href', '')
                    item_id = relative_url.split("/")[-1].split("?")[0] if "/" in relative_url else "unknown"
                    full_url = f"{self.base_url}/contenidos/ficha/{item_id}"
                    
                    # ENTRAR AL LINK /ficha/
                    ficha_data = self._scrape_ficha(full_url)
                    
                    # Metadata desde meta tags o fallback
                    authors = ficha_data.get('authors', [])
                    if not authors:
                        authors = ["UNAM Research Group"]
                        
                    doc = {
                        "id": item_id,
                        "title": ficha_data.get('title', title_elem.get_text(strip=True)),
                        "authors": authors,
                        "year": ficha_data.get('year', ""),
                        "abstract": ficha_data.get('abstract', ""),
                        "link": full_url,
                        "topics": [area_id, "UNAM Repository"]
                    }
                    self._process_document(doc)
                    processed_in_page += 1
                
                time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error scraping UNAM repository: {e}")
                break
                
    def _scrape_ficha(self, url: str) -> Dict:
        """Extrae metadata desde meta tags: author, citation_date, og:title, description."""
        ficha_data = {}
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=12)
            response.encoding = 'utf-8'
            if response.status_code != 200:
                return ficha_data
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraer metadata desde meta tags
            # author
            author_metas = soup.find_all('meta', attrs={'name': 'author'})
            if not author_metas:
                author_metas = soup.find_all('meta', attrs={'name': 'citation_author'})
            
            authors = [m.get('content') for m in author_metas if m.get('content')]
            if authors:
                ficha_data['authors'] = authors
                
            # citation_date
            date_meta = soup.find('meta', attrs={'name': 'citation_date'})
            if date_meta and date_meta.get('content'):
                date_str = date_meta.get('content')
                year_match = re.search(r"(\d{4})", date_str)
                ficha_data['year'] = year_match.group(1) if year_match else date_str[:4]
                
            # og:title
            title_meta = soup.find('meta', attrs={'property': 'og:title'})
            if title_meta:
                ficha_data['title'] = title_meta.get('content', '')
                
            # description
            desc_meta = soup.find('meta', attrs={'name': 'description'})
            if desc_meta:
                ficha_data['abstract'] = desc_meta.get('content', '')

            return ficha_data
        except Exception as e:
            logger.error(f"Error scraping ficha {url}: {e}")
            return ficha_data

    def _process_document(self, doc: Dict):
        """Procesa un documento para generar la estructura final de autores y trabajos."""
        title = doc.get("title", "Sin título")
        authors = doc.get("authors", ["UNAM Research Group"])
        year = doc.get("year", "")
        abstract = doc.get("abstract", "")
        link = doc.get("link", "")
        topics = doc.get("topics", [])
        item_id = doc.get("id", "unknown")
        
        self.processed_count += 1
        
        # Procesar autores para unam_authors.json
        for name in authors:
            name_norm = self.normalizer.normalize(name)
            if name_norm not in self.authors:
                self.authors[name_norm] = {
                    "id": f"unam_{name_norm.replace(' ', '_')}",
                    "nombre_completo": name,
                    "normalized_name": name_norm,
                    "institucion": "Universidad Nacional Autónoma de México",
                    "topics": topics,
                    "publications": [title],
                    "years": [year] if year else []
                }
            else:
                auth = self.authors[name_norm]
                if title not in auth["publications"]:
                    auth["publications"].append(title)
                if year and year not in auth["years"]:
                    auth["years"].append(year)

        # Guardar trabajo para unam_works.json
        self.works.append({
            "id": item_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "link": link,
            "topics": topics
        })

    def save(self):
        """Guarda los resultados en los archivos JSON correspondientes."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        authors_list = list(self.authors.values())
        with open(PROCESSED_DATA_DIR / "unam_authors.json", "w", encoding="utf-8") as f:
            json.dump(authors_list, f, indent=2, ensure_ascii=False)
            
        with open(PROCESSED_DATA_DIR / "unam_works.json", "w", encoding="utf-8") as f:
            json.dump(self.works, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✅ Scraping finalizado. Autores: {len(authors_list)}, Trabajos: {len(self.works)}")
