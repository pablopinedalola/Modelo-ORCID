"""
unam_directory_loader.py — Recolector minimalista para el directorio UNAM.

Implementación real (sin mocks) basada en web scraping de resultados públicos
para localizar e inferir el departamento y líneas de investigación de perfiles UNAM.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse

import requests

from config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class UNAMDirectoryLoader:
    """Carga y enriquece perfiles de investigadores de la UNAM mediante web scraping."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
        self.output_file = self.output_dir / "unam_directory.json"
        self.profiles: Dict[str, Dict[str, Any]] = {}
        
        # Cabeceras para simular un navegador real y evitar bloqueos básicos
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3",
        }

    def load(self) -> Dict[str, Dict[str, Any]]:
        """Carga los perfiles procesados desde el disco si existen."""
        if self.output_file.exists():
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
                logger.info(f"Cargados {len(self.profiles)} perfiles UNAM desde {self.output_file}")
            except Exception as e:
                logger.error(f"Error cargando perfiles UNAM: {e}")
        return self.profiles

    def save(self) -> None:
        """Persiste los perfiles extraídos en formato JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, ensure_ascii=False, indent=2)
            logger.info(f"Guardados {len(self.profiles)} perfiles en {self.output_file}")
        except Exception as e:
            logger.error(f"Error guardando perfiles UNAM: {e}")

    def fetch_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Busca un perfil en la UNAM utilizando DuckDuckGo HTML Search.
        Retorna la URL, departamento inferido y líneas de investigación desde el snippet.
        """
        logger.info(f"Buscando perfil UNAM para: {name}")
        
        # Construimos la query forzando el dominio de la UNAM
        query = f'"{name}" site:unam.mx (directorio OR investigador OR perfil)'
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            html = response.text
            
            # Simple regex to extract search results
            result_pattern = re.compile(r'<div class="result[^>]*>(.*?)</div></div>', re.DOTALL | re.IGNORECASE)
            url_pattern = re.compile(r'<a class="result__url" href="([^"]+)"')
            title_pattern = re.compile(r'<a class="result__title"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
            snippet_pattern = re.compile(r'<a class="result__snippet[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
            
            results = result_pattern.findall(html)
            
            for res_html in results:
                url_match = url_pattern.search(res_html)
                if not url_match:
                    continue
                    
                link = url_match.group(1).strip()
                # Clean duckduckgo redirect URL
                if "uddg=" in link:
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])
                elif link.startswith("//"):
                    link = "https:" + link
                
                # Ignorar PDFs o documentos, buscar páginas de perfiles
                if link.endswith(".pdf") or link.endswith(".doc"):
                    continue
                
                title_match = title_pattern.search(res_html)
                title = ""
                if title_match:
                    title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                
                snippet_match = snippet_pattern.search(res_html)
                snippet = ""
                if snippet_match:
                    snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                
                # Inferencia heurística básica para el departamento
                departamento = "UNAM"
                if "Facultad" in title:
                    departamento = "Facultad" + title.split("Facultad")[1].split("-")[0].strip()
                elif "Instituto" in title:
                    departamento = "Instituto" + title.split("Instituto")[1].split("-")[0].strip()
                elif "Facultad" in snippet:
                    departamento = "Facultad" + snippet.split("Facultad")[1].split(".")[0].split(",")[0].strip()
                elif "Instituto" in snippet:
                    departamento = "Instituto" + snippet.split("Instituto")[1].split(".")[0].split(",")[0].strip()
                
                if len(departamento) > 60:
                    departamento = "UNAM"  # Fallback si extrajo demasiado texto
                
                profile_data = {
                    "nombre": name,
                    "afiliacion": "Universidad Nacional Autónoma de México",
                    "departamento": departamento,
                    "lineas_de_investigacion": snippet,
                    "url_fuente": link
                }
                
                # Guardamos normalizando el nombre como key
                norm_name = self._normalize_name(name)
                self.profiles[norm_name] = profile_data
                logger.info(f"  Encontrado: {link}")
                
                return profile_data
                
            logger.warning(f"  No se encontraron resultados válidos para: {name}")
            return None
            
        except Exception as e:
            logger.error(f"Error realizando web scraping para {name}: {e}")
            return None

    def fetch_batch(self, names: List[str], delay_sec: float = 2.0) -> None:
        """Ejecuta la búsqueda para múltiples nombres respetando un delay."""
        for name in names:
            norm_name = self._normalize_name(name)
            if norm_name in self.profiles:
                logger.info(f"Perfil de {name} ya existe en caché, omitiendo.")
                continue
                
            self.fetch_profile(name)
            self.save()
            time.sleep(delay_sec)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normaliza el nombre para usarlo como llave en el diccionario."""
        import unicodedata
        name = name.strip().lower()
        nfkd = unicodedata.normalize("NFKD", name)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = UNAMDirectoryLoader()
    loader.load()
    
    test_names = [
        "Humberto Carrillo Calvet",
        "Miguel Alcubierre"
    ]
    loader.fetch_batch(test_names)
