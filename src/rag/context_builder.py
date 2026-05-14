"""
context_builder.py -- Construye el string de contexto para el LLM respetando límites de tokens.
"""

import logging
from typing import Any, Dict, List

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logging.warning("tiktoken no instalado. ContextBuilder usará heurísticas de caracteres.")

from config import CHAT_MAX_TOKENS
from src.rag.citation_builder import CitationBuilder

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Empaqueta chunks en un bloque de contexto para el prompt."""

    def __init__(self, max_tokens: int = CHAT_MAX_TOKENS):
        self.max_tokens = max_tokens
        if HAS_TIKTOKEN:
            self.encoding = tiktoken.get_encoding("cl100k_base") # Usado por gpt-4 y gpt-3.5

    def _count_tokens(self, text: str) -> int:
        if HAS_TIKTOKEN:
            return len(self.encoding.encode(text))
        return len(text) // 4  # Heurística simple

    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Construye el string final concatenando los resultados hasta el budget límite."""
        context_str = "CONTEXTO RECUPERADO DE LA BASE ACADÉMICA:\n\n"
        current_tokens = self._count_tokens(context_str)

        for i, res in enumerate(results):
            chunk_str = CitationBuilder.format_chunk_for_prompt(res, i) + "\n"
            chunk_tokens = self._count_tokens(chunk_str)
            
            if current_tokens + chunk_tokens > self.max_tokens:
                logger.info(f"Presupuesto de tokens alcanzado. Excluyendo resultado {i+1} en adelante.")
                break
                
            context_str += chunk_str
            current_tokens += chunk_tokens

        return context_str
