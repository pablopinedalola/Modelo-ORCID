"""
citation_builder.py -- Construye citas en formato académico para el LLM.
"""

from typing import Any, Dict


class CitationBuilder:
    """Genera strings de citas estandarizadas."""

    @staticmethod
    def create_citation(source_type: str, source_id: str, label: str) -> str:
        """Crea un tag de cita. Ej: [SNII:1234]"""
        return f"[{source_type}:{source_id} - {label}]"

    @staticmethod
    def format_chunk_for_prompt(chunk: Dict[str, Any], index: int) -> str:
        """Formatea un chunk recuperado para inyectarlo en el prompt."""
        cid = chunk.get("id", f"UNK_{index}")
        name = chunk.get("name", "Unknown")
        texts = chunk.get("chunks", [])
        
        text_block = "\n".join(f"- {t}" for t in texts)
        
        return f"=== REFERENCIA {index+1} [{cid}] ===\nInvestigador: {name}\nEvidencia:\n{text_block}\n"
