"""
query_interpreter.py — Motor de Natural Language Retrieval.

Funcionalidades:
1. Normalización de instituciones (UNAM, IPN, Tec, etc.)
2. Expansión conceptual automática (español ↔ inglés, acrónimos).
3. Detección de intención (authors, topics, papers).
4. Extracción de entidades y reescritura de queries.
"""

import re
import unicodedata
from typing import Dict, List, Set, Any

# =====================================================================
# 1. INSTITUTION NORMALIZATION
# =====================================================================
INSTITUTIONS = {
    "unam": ["universidad nacional autonoma de mexico", "unam", "instituto de astronomia unam", "iimas"],
    "ipn": ["instituto politecnico nacional", "ipn", "politecnico"],
    "cinvestav": ["centro de investigacion y de estudios avanzados", "cinvestav"],
    "uam": ["universidad autonoma metropolitana", "uam"],
    "tec": ["tecnologico de monterrey", "itesm", "tec de monterrey"],
    "udg": ["universidad de guadalajara", "udg"]
}

# =====================================================================
# 2. TOPIC EXPANSION & TRANSLATION DICTIONARY
# =====================================================================
CONCEPT_EXPANSION = {
    "ia": ["artificial intelligence", "machine learning", "deep learning", "inteligencia artificial", "ai"],
    "machine learning": ["artificial intelligence", "deep learning", "aprendizaje automatico", "ml"],
    "redes neuronales": ["neural networks", "deep learning", "artificial intelligence"],
    "teoria de graficas": ["graph theory", "complex networks", "combinatorics", "redes complejas"],
    "graficas": ["graph theory", "graphs", "redes", "networks"],
    "complejidad": ["complexity", "complex systems", "sistemas complejos"],
    "redes": ["networks", "complex networks", "graph theory"],
    "superconductividad": ["superconductivity", "condensed matter", "superconductors", "quantum materials"],
    "materiales": ["materials science", "condensed matter", "ciencia de materiales"]
}

class QueryInterpreter:
    """Interpreta queries en lenguaje natural y genera una estrategia de búsqueda."""

    def __init__(self):
        # Flatten institutions for easy lookup
        self.inst_map = {}
        for canonical, aliases in INSTITUTIONS.items():
            for alias in aliases:
                self.inst_map[alias] = canonical

    @staticmethod
    def _normalize(text: str) -> str:
        if not text: return ""
        text = text.lower().strip()
        # Quitar acentos
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in nfkd if not unicodedata.combining(c))
        # Quitar caracteres especiales excepto espacios y letras/números
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Quitar dobles espacios
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def interpret(self, raw_query: str) -> Dict[str, Any]:
        """
        Interpreta la query y retorna el plan de búsqueda:
        - intent: author|topic|paper|mixed
        - institutions: lista de instituciones detectadas
        - expanded_concepts: conceptos expandidos
        - rewritten_query: query optimizada para dense retrieval
        """
        norm_query = self._normalize(raw_query)
        words = norm_query.split()
        
        detected_institutions = set()
        expanded_concepts = set()
        intent = "mixed"

        # 1. Detect Intent
        if any(w in norm_query for w in ["investigador", "investigadores", "experto", "expertos", "autor", "autores"]):
            intent = "author"
        elif any(w in norm_query for w in ["paper", "papers", "articulo", "articulos", "publicacion"]):
            intent = "paper"
            
        # Limpiar stop words de la query de intención
        stop_words = ["investigador", "investigadores", "experto", "expertos", "autor", "autores",
                      "paper", "papers", "articulo", "articulos", "sobre", "en", "de", "del", "la", "las", "los", "que", "trabajen"]
        
        clean_words = [w for w in words if w not in stop_words]
        clean_text = " ".join(clean_words)

        # 2. Detect Institutions
        for alias, canonical in self.inst_map.items():
            if alias in clean_text:
                detected_institutions.add(canonical)
                # Remove institution from text to isolate concepts
                clean_text = clean_text.replace(alias, "").strip()

        # 3. Detect and Expand Concepts
        clean_words = clean_text.split()
        for i in range(len(clean_words)):
            # Try bigrams first
            if i < len(clean_words) - 1:
                bigram = f"{clean_words[i]} {clean_words[i+1]}"
                if bigram in CONCEPT_EXPANSION:
                    expanded_concepts.update(CONCEPT_EXPANSION[bigram])
            
            # Unigrams
            unigram = clean_words[i]
            if unigram in CONCEPT_EXPANSION:
                expanded_concepts.update(CONCEPT_EXPANSION[unigram])
            else:
                if len(unigram) > 3: # Keep meaningful words
                    expanded_concepts.add(unigram)

        if not expanded_concepts and clean_text:
            expanded_concepts.add(clean_text)

        # 4. Rewritten Query for Embedding
        rewritten_query = " ".join(expanded_concepts)
        if not rewritten_query:
            rewritten_query = raw_query

        return {
            "raw_query": raw_query,
            "normalized_query": norm_query,
            "intent": intent,
            "institutions": list(detected_institutions),
            "expanded_concepts": list(expanded_concepts),
            "rewritten_query": rewritten_query
        }
