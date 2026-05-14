"""
profile_builder.py — Generador de perfiles básicos a partir de registros SNII.

Responsabilidades:
    - Tomar una lista de SNIIRecord
    - Construir perfiles básicos con campos normalizados
    - Generar searchable_text concatenado para búsqueda futura
    - Exportar como lista de diccionarios listos para JSON/CSV

Este módulo NO implementa:
    - FAISS / embeddings
    - Retrieval híbrido
    - OpenAI / LLMs
    - Graph traversal

Es la base mínima sobre la cual se construirán las etapas posteriores.

Examples:
    >>> from src.loader.snii_loader import SNIILoader
    >>> from src.profiles.profile_builder import ProfileBuilder
    >>> loader = SNIILoader()
    >>> records = loader.load("data/raw/snii_sample.xlsx")
    >>> builder = ProfileBuilder()
    >>> profiles = builder.build_profiles(records)
    >>> print(profiles[0]["searchable_text"])
    'CARLOS ALBERTO GARCIA LOPEZ UNAM FISICA DE PARTICULAS ...'
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Optional

from src.models.schemas import SNIIRecord, SNIILevel

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# ÁREAS CONAHCyT
# ═════════════════════════════════════════════════════════════════════════════

AREA_LABELS = {
    "I": "Físico-Matemáticas y Ciencias de la Tierra",
    "II": "Biología y Química",
    "III": "Ciencias Médicas y de la Salud",
    "IV": "Humanidades y Ciencias de la Conducta",
    "V": "Ciencias Sociales",
    "VI": "Biotecnología y Ciencias Agropecuarias",
    "VII": "Ingenierías",
}

NIVEL_LABELS = {
    "C": "Candidato",
    "1": "SNI I",
    "2": "SNI II",
    "3": "SNI III",
    "E": "Emérito",
    "?": "No especificado",
}


class ProfileBuilder:
    """Construye perfiles básicos de investigador a partir de registros SNII.

    Genera un diccionario por investigador con campos normalizados y un
    campo `searchable_text` que concatena toda la información relevante
    para facilitar búsquedas futuras (fuzzy, full-text o semánticas).

    Attributes:
        profiles_built: Contador de perfiles generados.

    Examples:
        >>> builder = ProfileBuilder()
        >>> profiles = builder.build_profiles(records)
        >>> len(profiles)
        1000
    """

    def __init__(self) -> None:
        self.profiles_built: int = 0

    def build_profiles(self, records: list[SNIIRecord]) -> list[dict]:
        """Construye perfiles básicos para todos los registros.

        Args:
            records: Lista de SNIIRecord cargados y limpios.

        Returns:
            Lista de diccionarios con perfil básico de cada investigador.
        """
        profiles = []
        errors = 0

        for record in records:
            try:
                profile = self.build_single_profile(record)
                profiles.append(profile)
            except Exception as e:
                errors += 1
                logger.warning(f"⚠️  Error generando perfil para {record.full_name}: {e}")
                continue

        self.profiles_built = len(profiles)

        if errors > 0:
            logger.warning(f"⚠️  {errors} registros con error durante la generación de perfiles")

        return profiles

    def build_single_profile(self, record: SNIIRecord) -> dict:
        """Construye el perfil básico de un solo investigador.

        Args:
            record: Registro SNII individual.

        Returns:
            Diccionario con el perfil básico:
                - id: Identificador único
                - nombre_completo: Nombre completo normalizado
                - nombre: Nombre(s) de pila
                - paterno: Apellido paterno
                - materno: Apellido materno
                - institucion: Institución de adscripción
                - dependencia: Dependencia / departamento
                - subdependencia: Subdependencia
                - area: Código de área CONAHCyT
                - area_nombre: Nombre completo del área
                - disciplina: Disciplina específica
                - nivel: Código del nivel SNI
                - nivel_nombre: Nombre del nivel SNI
                - searchable_text: Texto concatenado para búsqueda

        Examples:
            >>> profile = builder.build_single_profile(record)
            >>> profile["nombre_completo"]
            'CARLOS ALBERTO GARCIA LOPEZ'
        """
        # Construir nombre completo
        nombre_completo = record.full_name

        # Resolver labels
        area_nombre = AREA_LABELS.get(record.area.strip().upper(), record.area)
        nivel_nombre = NIVEL_LABELS.get(record.nivel.value, record.nivel.value)

        # Construir searchable_text
        searchable_text = self._build_searchable_text(
            nombre_completo=nombre_completo,
            institucion=record.institucion,
            dependencia=record.dependencia,
            subdependencia=record.subdependencia,
            area=record.area,
            area_nombre=area_nombre,
            disciplina=record.disciplina,
            nivel_nombre=nivel_nombre,
        )

        return {
            "id": record.id,
            "nombre_completo": nombre_completo,
            "nombre": record.nombre,
            "paterno": record.paterno,
            "materno": record.materno,
            "institucion": record.institucion,
            "dependencia": record.dependencia,
            "subdependencia": record.subdependencia,
            "area": record.area,
            "area_nombre": area_nombre,
            "disciplina": record.disciplina,
            "nivel": record.nivel.value,
            "nivel_nombre": nivel_nombre,
            "searchable_text": searchable_text,
        }

    def _build_searchable_text(
        self,
        nombre_completo: str,
        institucion: str,
        dependencia: str,
        subdependencia: str,
        area: str,
        area_nombre: str,
        disciplina: str,
        nivel_nombre: str,
    ) -> str:
        """Construye el texto searchable concatenando toda la info relevante.

        El searchable_text es clave para las etapas posteriores del pipeline:
        - Búsqueda fuzzy rápida
        - Generación de embeddings
        - Full-text search

        Se normaliza removiendo acentos y pasando a lowercase para
        maximizar la compatibilidad con diferentes métodos de búsqueda.

        Args:
            nombre_completo: Nombre completo del investigador.
            institucion: Institución de adscripción.
            dependencia: Dependencia.
            subdependencia: Subdependencia.
            area: Código de área.
            area_nombre: Nombre del área.
            disciplina: Disciplina.
            nivel_nombre: Nombre del nivel SNI.

        Returns:
            Texto concatenado, normalizado (sin acentos, lowercase).
        """
        parts = [
            nombre_completo,
            institucion,
            dependencia,
            subdependencia,
            area,
            area_nombre,
            disciplina,
            nivel_nombre,
        ]

        # Filtrar vacíos, unir y normalizar
        raw_text = " ".join(p.strip() for p in parts if p and p.strip())
        return self._normalize_text(raw_text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normaliza texto removiendo acentos y caracteres especiales.

        Args:
            text: Texto a normalizar.

        Returns:
            Texto en lowercase, sin acentos, con espacios simples.

        Examples:
            >>> ProfileBuilder._normalize_text("GARCÍA LÓPEZ  Física")
            'garcia lopez fisica'
        """
        # Lowercase
        text = text.lower()

        # Remover acentos / diacríticos
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in nfkd if not unicodedata.combining(c))

        # Limpiar espacios múltiples
        text = " ".join(text.split())

        return text

    def summary(self, profiles: list[dict]) -> dict:
        """Genera un resumen estadístico de los perfiles.

        Args:
            profiles: Lista de perfiles generados.

        Returns:
            Diccionario con estadísticas de cobertura de campos.
        """
        total = len(profiles)
        if total == 0:
            return {"total": 0}

        coverage = {}
        for field in ["nombre_completo", "institucion", "area", "disciplina", "nivel"]:
            present = sum(1 for p in profiles if p.get(field) and p[field] not in ("", "?"))
            coverage[field] = {
                "presentes": present,
                "porcentaje": round(present / total * 100, 1),
            }

        # Longitud promedio de searchable_text
        avg_len = sum(len(p.get("searchable_text", "")) for p in profiles) / total

        return {
            "total_perfiles": total,
            "cobertura_campos": coverage,
            "longitud_promedio_searchable_text": round(avg_len, 1),
        }
