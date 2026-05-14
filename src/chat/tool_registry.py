"""
tool_registry.py -- Definición de herramientas para el LLM.
"""

import json
from typing import Any, Callable, Dict, List

from src.rag.retriever import HybridRetriever
from src.rag.graph_retriever import GraphRetriever

class ToolRegistry:
    """Registra y ejecuta tools invocables por el LLM."""

    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.graph_retriever = GraphRetriever()
        
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "search_academic_context",
                    "description": "Busca contexto en la base de datos RAG usando búsqueda semántica e híbrida. Útil para consultas como 'quién trabaja en X' o 'busca a Y'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "La consulta de búsqueda académica."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_graph_neighborhood",
                    "description": "Explora el grafo académico para obtener vecinos y coautores de un investigador específico.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "researcher_id": {
                                "type": "string",
                                "description": "El ID del investigador (ej. 'snii:1234' o '1234')."
                            }
                        },
                        "required": ["researcher_id"]
                    }
                }
            }
        ]
        
    def execute_tool(self, name: str, arguments: str, session_context: str = None) -> str:
        """Ejecuta la tool correspondiente y devuelve el resultado en JSON."""
        try:
            args = json.loads(arguments)
            
            if name == "search_academic_context":
                return self._search_academic_context(args.get("query"), session_context)
            elif name == "get_graph_neighborhood":
                return self._get_graph_neighborhood(args.get("researcher_id"))
            else:
                return json.dumps({"error": f"Tool {name} no encontrada."})
                
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _search_academic_context(self, query: str, context_id: str = None) -> str:
        results = self.hybrid_retriever.search(query, context_id=context_id, top_k=5)
        # Formateamos resultados de forma condensada para el LLM
        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("id"),
                "name": r.get("name"),
                "sources": r.get("sources"),
                "relevant_chunks": r.get("chunks", [])[:3]
            })
        return json.dumps(formatted, ensure_ascii=False)

    def _get_graph_neighborhood(self, researcher_id: str) -> str:
        neighbors = self.graph_retriever.get_neighborhood(researcher_id)
        if not neighbors:
            return json.dumps({"message": "No se encontraron vecinos en el grafo."})
            
        formatted = [{"id": n.get("id"), "label": n.get("label"), "type": n.get("type")} for n in neighbors[:15]]
        return json.dumps({"neighbors": formatted}, ensure_ascii=False)
