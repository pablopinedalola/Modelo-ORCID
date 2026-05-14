"""
conversation_memory.py -- Gestión de sesiones y contexto de la conversación.
"""

from typing import Any, Dict, List, Optional


class ConversationMemory:
    """Mantiene el historial de mensajes por sesión."""

    def __init__(self):
        # Mapeo session_id -> list of messages
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        # Para resolución de coreferencias (ej. "hablame de el")
        self.active_context_ids: Dict[str, str] = {} 

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {"role": "system", "content": "Eres el Asistente Académico Modelo-ORCID (tipo OpenAlex/Wikipedia de México). Usas herramientas para explorar el grafo y recuperar contexto. SIEMPRE debes citar tus fuentes."}
            ]
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        
        # Limitar historial para no romper contexto (últimos 15 mensajes)
        if len(history) > 15:
            # Mantener system prompt
            self.sessions[session_id] = [history[0]] + history[-14:]

    def add_tool_message(self, session_id: str, tool_call_id: str, name: str, content: str):
        history = self.get_history(session_id)
        history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content
        })

    def set_active_context(self, session_id: str, researcher_id: str):
        """Guarda el ID del último investigador consultado."""
        self.active_context_ids[session_id] = researcher_id

    def get_active_context(self, session_id: str) -> Optional[str]:
        return self.active_context_ids.get(session_id)
