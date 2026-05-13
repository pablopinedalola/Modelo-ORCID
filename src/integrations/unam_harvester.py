"""
unam_harvester.py -- Recolector de metadatos de repositorios abiertos de México.

Implementa un cliente básico de OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting)
para extraer metadatos de publicaciones en formato Dublin Core (oai_dc) sin requerir scraping agresivo.
"""

import os
import json
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from config import DATA_DIR

logger = logging.getLogger(__name__)

class OAIHarvester:
    """Cliente básico OAI-PMH para extraer metadatos de repositorios abiertos."""

    OAI_NS = {
        'oai': 'http://www.openarchives.org/OAI/2.0/',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
    }

    def __init__(self, base_url: str, repository_name: str, output_dir: Path):
        self.base_url = base_url
        self.repository_name = repository_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def harvest(self, max_records: int = 100) -> List[Dict[str, Any]]:
        """Descarga metadatos usando el verbo ListRecords (metadataPrefix=oai_dc)."""
        logger.info(f"Iniciando harvest desde {self.repository_name} ({self.base_url})")
        
        params = {
            'verb': 'ListRecords',
            'metadataPrefix': 'oai_dc'
        }
        
        records_collected = []
        resumption_token = None
        
        while len(records_collected) < max_records:
            if resumption_token:
                params = {
                    'verb': 'ListRecords',
                    'resumptionToken': resumption_token
                }
                
            try:
                # OAI-PMH servers can be slow, adding timeout
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Error conectando a {self.base_url}: {e}")
                break
                
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                logger.error(f"Error parseando XML de {self.base_url}: {e}")
                break
            
            # Revisar si hay errores OAI (e.g. badArgument, cannotDisseminateFormat)
            error = root.find('oai:error', self.OAI_NS)
            if error is not None:
                logger.error(f"Error OAI: {error.text}")
                break

            # Procesar records
            for record_elem in root.findall('.//oai:record', self.OAI_NS):
                if len(records_collected) >= max_records:
                    break
                    
                header = record_elem.find('oai:header', self.OAI_NS)
                if header is not None and header.get('status') == 'deleted':
                    continue # Skip deleted records
                    
                metadata = record_elem.find('.//oai_dc:dc', self.OAI_NS)
                if metadata is not None:
                    parsed_record = self._parse_dc_metadata(metadata)
                    if parsed_record:
                        records_collected.append(parsed_record)
            
            # Checar resumption token para paginación
            token_elem = root.find('.//oai:resumptionToken', self.OAI_NS)
            if token_elem is not None and token_elem.text:
                resumption_token = token_elem.text
            else:
                break # No hay más páginas
                
        logger.info(f"Recolectados {len(records_collected)} registros de {self.repository_name}")
        self._save_records(records_collected)
        return records_collected

    def _parse_dc_metadata(self, dc_elem: ET.Element) -> Dict[str, Any]:
        """Extrae campos del XML Dublin Core a un diccionario."""
        record = {
            'title': '',
            'authors': [],
            'abstract': '',
            'keywords': [],
            'institution': self.repository_name,
            'publication_year': None,
            'url': '',
            'type': '',
            'language': ''
        }
        
        # Título
        title_elem = dc_elem.find('dc:title', self.OAI_NS)
        if title_elem is not None and title_elem.text:
            record['title'] = title_elem.text.strip()
            
        # Autores (creators)
        for creator in dc_elem.findall('dc:creator', self.OAI_NS):
            if creator.text:
                record['authors'].append(creator.text.strip())
                
        # Abstract / Descripción
        for desc in dc_elem.findall('dc:description', self.OAI_NS):
            if desc.text:
                record['abstract'] += desc.text.strip() + " "
        record['abstract'] = record['abstract'].strip()
        
        # Keywords / Materias
        for subject in dc_elem.findall('dc:subject', self.OAI_NS):
            if subject.text:
                record['keywords'].append(subject.text.strip())
                
        # Fecha / Año
        date_elem = dc_elem.find('dc:date', self.OAI_NS)
        if date_elem is not None and date_elem.text:
            date_str = date_elem.text.strip()
            # Extracción rudimentaria del año (primeros 4 dígitos)
            if len(date_str) >= 4 and date_str[:4].isdigit():
                record['publication_year'] = int(date_str[:4])
                
        # Identificador / URL
        for ident in dc_elem.findall('dc:identifier', self.OAI_NS):
            if ident.text and ('http://' in ident.text or 'https://' in ident.text):
                record['url'] = ident.text.strip()
                break
                
        # Tipo de documento
        type_elem = dc_elem.find('dc:type', self.OAI_NS)
        if type_elem is not None and type_elem.text:
            record['type'] = type_elem.text.strip()

        # Idioma
        lang_elem = dc_elem.find('dc:language', self.OAI_NS)
        if lang_elem is not None and lang_elem.text:
            record['language'] = lang_elem.text.strip()
            
        # Validar si tiene mínimo un título y autores para ser considerado válido
        return record if record['title'] and record['authors'] else None

    def _save_records(self, records: List[Dict[str, Any]]):
        """Persiste los registros en formato JSON."""
        if not records:
            return
            
        filename = f"{self.repository_name.lower().replace(' ', '_')}_harvest.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Guardados en {filepath}")

def run_unam_harvesting():
    """
    Función principal para ejecutar la recolección.
    Se conecta a repositorios abiertos como SciELO México y extrae una muestra.
    (Nota: las URLs de repositorios institucionales de UNAM pueden variar o requerir red interna,
    por lo que se usan endpoints OAI públicos estándar).
    """
    REPOSITORIES = [
        {
            "name": "SciELO Mexico",
            "url": "http://www.scielo.org.mx/oai/scielo-oai.php"
        },
        # Ejemplo de otros posibles endpoints:
        # {
        #     "name": "TesiUNAM",
        #     "url": "https://tesiunam.dgb.unam.mx/oai/request"
        # }
    ]
    
    out_dir = DATA_DIR / "unam"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    for repo in REPOSITORIES:
        harvester = OAIHarvester(repo['url'], repo['name'], out_dir)
        # Limitamos a 50 para prueba de concepto sin sobrecargar servidores
        records = harvester.harvest(max_records=50)
        all_records.extend(records)
        
    logger.info(f"Proceso completado. Total de registros recolectados: {len(all_records)}")
    return all_records

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_unam_harvesting()
