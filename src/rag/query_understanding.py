"""
query_understanding.py — Módulo de entendimiento avanzado de queries académicas.

Expande el query simple para incluir:
- Mapeo cross-lingual (Español <-> Inglés)
- Detección de disciplinas e intención
- Expansión de sinónimos académicos y equivalencias de OpenAlex
"""

import json
import logging
import unicodedata
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Diccionario bidireccional de sinónimos y traducciones académicas
ACADEMIC_DICTIONARY = {
    # Matemáticas y Computación
    "teoria de graficas": ["graph theory", "discrete optimization", "combinatorics", "redes complejas", "complex networks", "algoritmos"],
    "redes neuronales": ["neural networks", "deep learning", "artificial intelligence", "machine learning", "aprendizaje automatico"],
    "inteligencia artificial": ["artificial intelligence", "machine learning", "deep learning", "ai", "ml"],
    "aprendizaje automatico": ["machine learning", "pattern recognition", "deep learning"],
    
    # Física
    "superconductividad": ["superconductivity", "condensed matter physics", "quantum materials", "física del estado sólido", "superconductors"],
    "optica cuantica": ["quantum optics", "quantum mechanics", "física cuántica", "photonics"],
    "mecanica cuantica": ["quantum mechanics", "quantum physics", "física teórica"],
    "fisica de particulas": ["particle physics", "high energy physics", "física de altas energías"],

    # Ciencias de la Salud / Bio
    "medicina": ["medicine", "ciencias de la salud", "clinical medicine"],
    "bioinformatica": ["bioinformatics", "computational biology", "biología computacional"],
    "epidemiologia": ["epidemiology", "public health", "salud pública"],

    # Generales
    "algoritmos": ["algorithms", "computer science"],
    "robotica": ["robotics", "control theory", "mechatronics"],
}

# Invertir/expandir el diccionario de forma automática para bidireccionalidad simple
EXPANDED_DICT: Dict[str, Set[str]] = {}
for k, v_list in ACADEMIC_DICTIONARY.items():
    if k not in EXPANDED_DICT:
        EXPANDED_DICT[k] = set()
    for v in v_list:
        EXPANDED_DICT[k].add(v)
        if v not in EXPANDED_DICT:
            EXPANDED_DICT[v] = set()
        EXPANDED_DICT[v].add(k)


class QueryAnalyzer:
    def __init__(self):
        pass
        
    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def analyze(self, query: str) -> dict:
        """Analiza la consulta y determina intención, expansiones y posibles temas."""
        norm_query = self._normalize(query)
        
        expansions = set([norm_query])
        detected_topics = []
        
        # Simple n-gram matching (hasta 3-grams)
        words = norm_query.split()
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                if ngram in EXPANDED_DICT:
                    detected_topics.append(ngram)
                    expansions.update(EXPANDED_DICT[ngram])
        
        # Detección de intención
        intent = "mixed"
        if "universidad" in norm_query or "instituto" in norm_query or "center" in norm_query:
            intent = "institution search"
        elif "papers" in norm_query or "articulos" in norm_query or "sobre" in norm_query:
            intent = "paper search"
        elif "investigador" in norm_query or "autor" in norm_query or "profesor" in norm_query:
            intent = "author search"
        elif detected_topics:
            intent = "topic search"
            
        # Limpiar stop words de la expansión final
        stop_words = {"sobre", "de", "la", "el", "los", "las", "en", "para", "que", "papers", "investigadores", "autores"}
        clean_expansions = {e for e in expansions if e not in stop_words}
        
        return {
            "original_query": query,
            "normalized_query": norm_query,
            "intent": intent,
            "detected_topics": detected_topics,
            "expanded_terms": list(clean_expansions),
            "search_string": " ".join(clean_expansions)
        }
