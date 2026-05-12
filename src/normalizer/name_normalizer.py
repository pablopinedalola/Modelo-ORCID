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

        Examples:
            >>> aliases = NameNormalizer().generate_aliases("CARLOS", "GARCÍA", "LÓPEZ")
            >>> "García López, Carlos" in aliases
            True
        """
        aliases: list[str] = []
        seen: set[str] = set()

        def add(alias: str) -> None:
            clean = alias.strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                aliases.append(clean)

        parts = nombre.strip().split()
        pat = paterno.strip()
        mat = materno.strip()

        pat_na = self.remove_accents(pat)
        mat_na = self.remove_accents(mat)
        nombre_na = self.remove_accents(nombre.strip())

        initial = parts[0][0] + "." if parts else ""
        all_init = " ".join(p[0] + "." for p in parts) if parts else ""
        first = parts[0] if parts else ""

        n_tc = " ".join(w.capitalize() for w in parts)
        p_tc, m_tc = pat.capitalize(), mat.capitalize()
        n_na_tc = " ".join(w.capitalize() for w in nombre_na.split())
        p_na_tc = pat_na.capitalize()
        m_na_tc = mat_na.capitalize()
        f_tc = first.capitalize()
        f_na_tc = self.remove_accents(f_tc)

        full = f"{p_tc} {m_tc}".strip() if m_tc else p_tc
        full_na = f"{p_na_tc} {m_na_tc}".strip() if m_na_tc else p_na_tc

        # Nombre Apellido1 Apellido2
        add(f"{n_tc} {full}")
        add(f"{n_na_tc} {full_na}")
        # Inicial. Apellido
        add(f"{initial} {full}")
        add(f"{initial} {full_na}")
        # Apellido, Nombre
        add(f"{full}, {n_tc}")
        add(f"{full_na}, {n_na_tc}")
        # Apellido1-Apellido2 (guión)
        if m_tc:
            add(f"{p_tc}-{m_tc}, {n_tc}")
            add(f"{p_na_tc}-{m_na_tc}, {n_na_tc}")
            add(f"{n_tc} {p_tc}-{m_tc}")
            add(f"{n_na_tc} {p_na_tc}-{m_na_tc}")
        # Iniciales completas
        if len(parts) > 1:
            add(f"{all_init} {full}")
        # Solo primer apellido
        add(f"{f_tc} {p_tc}")
        add(f"{f_na_tc} {p_na_tc}")
        add(f"{initial} {p_tc}")
        add(f"{initial} {p_na_tc}")

        return aliases

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
