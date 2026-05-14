"""
chunker.py -- Academic Chunking para el RAG.

Divide perfiles académicos, publicaciones y evidencia en 
fragmentos (chunks) semánticos que el Vector Store puede indexar.
Evita dividir texto de forma arbitraria, manteniendo la cohesión.
"""

from typing import Any, Dict, List


class AcademicChunker:
    """Convierte datos académicos en chunks estructurados para RAG."""

    def __init__(self, max_length: int = 512):
        self.max_length = max_length

    def chunk_profile(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera chunks a partir de un perfil resuelto.
        
        Crea:
        1. Bio/Metadata Chunk
        2. Publications Chunk
        3. Concepts Chunk
        """
        chunks = []
        pid = profile.get("id", "unknown")
        name = profile.get("full_name", "Unknown")

        # 1. Bio Chunk
        bio_text = f"Investigador: {name}. "
        if profile.get("institution"):
            bio_text += f"Institución: {profile['institution']}. "
        if profile.get("department"):
            bio_text += f"Departamento: {profile['department']}. "
        if profile.get("discipline"):
            bio_text += f"Disciplina: {profile['discipline']}. "
        if profile.get("orcid_id"):
            bio_text += f"ORCID: {profile['orcid_id']}. "
        
        chunks.append({
            "text": bio_text.strip(),
            "metadata": {
                "researcher_id": pid,
                "type": "bio",
                "name": name
            }
        })

        # 2. Publications Chunk
        pubs = profile.get("publications", [])
        if pubs:
            # Agrupar títulos para no exceder max_length si es posible
            pub_texts = [p.get("title", "") for p in pubs if p.get("title")]
            if pub_texts:
                pub_chunk = f"Publicaciones de {name}: " + " | ".join(pub_texts[:10]) # Limitar a 10 para contexto
                chunks.append({
                    "text": pub_chunk,
                    "metadata": {
                        "researcher_id": pid,
                        "type": "publications",
                        "name": name
                    }
                })

        # 3. Concepts Chunk
        concepts = profile.get("concepts", [])
        if concepts:
            concept_chunk = f"Áreas de investigación y conceptos de {name}: " + ", ".join(concepts)
            chunks.append({
                "text": concept_chunk,
                "metadata": {
                    "researcher_id": pid,
                    "type": "concepts",
                    "name": name
                }
            })

        return chunks

    def chunk_explanation(self, researcher_id: str, name: str, explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera chunks a partir de la interpretabilidad (explainability)."""
        chunks = []
        if not explanation:
            return chunks

        text = f"Explicación de asignación para {name}: "
        text += f"Confianza: {explanation.get('confidence_score', 0):.2f}. "
        
        summary = explanation.get("evidence_summary", {})
        if summary.get("institution_match"):
            text += "La institución coincide. "
        if summary.get("topics_match"):
            text += "Los temas de investigación coinciden fuertemente. "
            
        chunks.append({
            "text": text,
            "metadata": {
                "researcher_id": researcher_id,
                "type": "explanation",
                "name": name
            }
        })
        return chunks
