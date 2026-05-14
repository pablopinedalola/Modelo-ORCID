"""
name_normalizer.py — Normalización de nombres de investigadores.

Transforma nombres del padrón SNII a formas normalizadas y genera
aliases para maximizar el matching contra ORCID y OpenAlex.

Examples:
    >>> norm = NameNormalizer()
    >>> norm.normalize("GARCÍA LÓPEZ")
    'garcia lopez'
    >>> aliases = norm.generate_aliases("CARLOS", "GARCÍA", "LÓPEZ")
    >>> "C. García López" in aliases
    True
"""

from __future__ import annotations

import re
import unicodedata
import logging

from src.models.schemas import SNIIRecord

logger = logging.getLogger(__name__)


class NameNormalizer:
    """Normaliza nombres de investigadores mexicanos y genera aliases.

    Maneja particularidades de nombres hispanohablantes:
    dos apellidos, nombres compuestos, acentos, ñ, y variaciones
    de publicación internacional.
    """

    TITLES = {
        "dr", "dra", "dr.", "dra.", "prof", "prof.",
        "ing", "ing.", "lic", "lic.", "mtro", "mtro.",
        "mtra", "mtra.", "phd", "ph.d.", "md", "m.d.",
    }

    NAME_PARTICLES = {"de", "del", "la", "las", "los", "y", "e"}

    def normalize(self, text: str) -> str:
        """Normaliza: lowercase, sin acentos, sin títulos.

        Examples:
            >>> NameNormalizer().normalize("Dr. GARCÍA")
            'garcia'
        """
        if not text:
            return ""
        text = text.strip().lower()
        text = self._remove_titles(text)
        text = self.remove_accents(text)
        text = re.sub(r"[^a-z\s\-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def remove_accents(text: str) -> str:
        """Remueve acentos preservando ñ como n."""
        text = text.replace("ñ", "n").replace("Ñ", "N")
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _remove_titles(self, text: str) -> str:
        words = text.split()
        cleaned = [w for w in words if w.lower().rstrip(".") not in self.TITLES
                    and w.lower() not in self.TITLES]
        return " ".join(cleaned) if cleaned else text

    def generate_aliases(self, nombre: str, paterno: str, materno: str = "") -> list[str]:
        """Genera variaciones del nombre para búsqueda en APIs.

        Produce formatos comunes en publicaciones internacionales.

        Args:
            nombre: Nombre(s) de pila.
            paterno: Apellido paterno.
            materno: Apellido materno (opcional).

        Returns:
            Lista de aliases únicos.
        """
        aliases: list[str] = []
        seen: set[str] = set()

        def add(alias: str) -> None:
            clean = alias.strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                aliases.append(clean)

        nombre = nombre.strip()
        pat = paterno.strip()
        mat = materno.strip()
        
        parts = nombre.split()
        first_name = parts[0] if parts else ""
        initial = first_name[0] + "." if first_name else ""
        all_init = " ".join(p[0] + "." for p in parts) if parts else ""

        # Normalizados (sin acentos)
        nombre_na = self.remove_accents(nombre)
        pat_na = self.remove_accents(pat)
        mat_na = self.remove_accents(mat)

        # 1. Formatos completos
        add(f"{nombre} {pat} {mat}")
        add(f"{nombre_na} {pat_na} {mat_na}")
        add(f"{pat} {mat}, {nombre}")
        add(f"{pat_na} {mat_na}, {nombre_na}")

        # 2. Formatos con iniciales
        add(f"{initial} {pat} {mat}")
        add(f"{initial} {pat_na} {mat_na}")
        if all_init:
            add(f"{all_init} {pat}")
            add(f"{all_init} {pat_na}")

        # 3. Formatos de publicación internacional (Surname, F. or Surname, First)
        add(f"{pat}, {nombre}")
        add(f"{pat_na}, {nombre_na}")
        add(f"{pat}, {initial}")
        add(f"{pat_na}, {initial}")
        
        # 4. Formatos con guión (común en México para preservar ambos apellidos)
        if mat:
            add(f"{pat}-{mat}, {nombre}")
            add(f"{pat_na}-{mat_na}, {nombre_na}")
            add(f"{nombre} {pat}-{mat}")
            add(f"{nombre_na} {pat_na}-{mat_na}")

        # 5. Formatos cortos (solo primer apellido)
        add(f"{first_name} {pat}")
        add(self.remove_accents(f"{first_name} {pat}"))
        
        # 6. Variantes de guiones (Carrillo Calvet <-> Carrillo-Calvet)
        if mat:
            add(f"{pat} {mat}")
            add(f"{pat}-{mat}")
            add(f"{nombre} {pat} {mat}")
            add(f"{nombre} {pat}-{mat}")
            add(f"{initial} {pat}-{mat}")
            
        # 7. Solo apellidos
        add(pat)
        if mat:
            add(mat)
            add(f"{pat} {mat}")
            add(f"{pat}-{mat}")
            
        return list(seen) # Retornar las formas normalizadas para facilitar el matching exacto

    def normalize_query(self, query: str) -> str:
        """Normalización fuerte para matching de queries."""
        return self.normalize(query)

    def tokenize(self, text: str) -> set[str]:
        """Extrae tokens significativos para matching.

        Examples:
            >>> NameNormalizer().tokenize("Carlos de García")
            {'carlos', 'garcia'}
        """
        normalized = self.normalize(text)
        return {t for t in normalized.split()
                if t not in self.NAME_PARTICLES and len(t) > 1}

    def normalize_record(self, record: SNIIRecord) -> dict:
        """Normaliza un registro SNII completo.

        Returns:
            Dict con normalized_name, aliases, tokens.
        """
        normalized_name = self.normalize(record.full_name)
        aliases = self.generate_aliases(record.nombre, record.paterno, record.materno)
        tokens = self.tokenize(record.full_name)
        logger.debug(f"  '{record.full_name}' → '{normalized_name}' ({len(aliases)} aliases)")
        return {"normalized_name": normalized_name, "aliases": aliases, "tokens": tokens}
