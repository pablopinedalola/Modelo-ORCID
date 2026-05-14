"""
chat_engine.py -- Capa conversacional RAG simple usando Ollama.
"""

import json
import logging
import requests
from typing import Dict, Any, Tuple, List

from src.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

class AcademicChatEngine:
    """Motor conversacional académico ligero basado en Ollama local."""
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5:7b"
        
    def process_message(self, message: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Recibe la pregunta, recupera contexto, y genera respuesta con Ollama.
        Retorna la respuesta y la lista de fuentes.
        """
        # 1. Usar HybridRetriever existente para obtener top_k=5 documentos
        try:
            results = self.retriever.search(query=message, top_k=5)
        except Exception as e:
            logger.error(f"Error en retriever: {e}")
            results = []
            
        # 2. Construir contexto textual
        context_parts = []
        sources = []
        
        for res in results:
            text = res.get("text", "")
            meta = res.get("metadata", {})
            
            name = meta.get("name", "Documento desconocido")
            title = meta.get("title", name)
            
            context_parts.append(f"--- Documento/Investigador: {title} ---\n{text}")
            
            sources.append({
                "title": title,
                "score": res.get("score", 0.0),
                "id": meta.get("id", "")
            })
            
        context_text = "\n\n".join(context_parts)
        
        # 3. Prompt con reglas estrictas
        prompt = f"""Eres un asistente académico experto de un sistema de exploración de investigadores.
REGLAS ESTRICTAS:
- Responde SOLO utilizando la información proporcionada en el CONTEXTO RECUPERADO.
- Cita los títulos o nombres de investigadores explícitamente en tu respuesta si los utilizas.
- Si la información en el contexto no es suficiente para responder la pregunta, debes admitir tu incertidumbre (di que no tienes información suficiente).
- NUNCA inventes papers, autores, instituciones ni hechos que no estén en el contexto.

CONTEXTO RECUPERADO:
{context_text}

PREGUNTA DEL USUARIO:
{message}

RESPUESTA (en español, clara y profesional):
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        # 4. Llamar Ollama vía HTTP requests
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "No se obtuvo respuesta del modelo.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error llamando a Ollama: {e}")
            answer = ("Lo siento, el servicio de generación no está disponible en este momento "
                      "o el modelo local no está respondiendo.")
            
        return answer, sources
