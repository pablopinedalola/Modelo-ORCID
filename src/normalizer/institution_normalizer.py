"""
institution_normalizer.py — Normalización de instituciones mexicanas.

Mapea nombres de instituciones del SNII a formas canónicas,
genera aliases (abreviaturas, nombres en inglés) y prepara
queries para la API de ROR.

Examples:
    >>> norm = InstitutionNormalizer()
    >>> norm.normalize("UNIVERSIDAD NACIONAL AUTONOMA DE MEXICO")
    'Universidad Nacional Autónoma de México'
    >>> norm.get_aliases("UNAM")
    ['Universidad Nacional Autónoma de México', 'UNAM',
     'National Autonomous University of Mexico']
"""

from __future__ import annotations

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InstitutionNormalizer:
    """Normaliza instituciones académicas mexicanas.

    Mantiene un diccionario de ~50 instituciones principales del
    sistema de investigación mexicano con sus aliases, abreviaturas,
    nombres en inglés y ROR IDs conocidos.
    """

    # ═══════════════════════════════════════════════════════════════════
    # DICCIONARIO DE INSTITUCIONES MEXICANAS
    # Estructura: canonical_name → {abbr, english, ror_id, aliases}
    # ═══════════════════════════════════════════════════════════════════

    INSTITUTIONS: dict[str, dict] = {
        "Universidad Nacional Autónoma de México": {
            "abbr": "UNAM",
            "english": "National Autonomous University of Mexico",
            "ror_id": "https://ror.org/01tmp8f25",
            "aliases": [
                "UNIVERSIDAD NACIONAL AUTONOMA DE MEXICO",
                "UNAM", "U.N.A.M.",
            ],
        },
        "Instituto Politécnico Nacional": {
            "abbr": "IPN",
            "english": "National Polytechnic Institute",
            "ror_id": "https://ror.org/059sp8j34",
            "aliases": [
                "INSTITUTO POLITECNICO NACIONAL",
                "IPN", "I.P.N.",
            ],
        },
        "Centro de Investigación y de Estudios Avanzados del IPN": {
            "abbr": "CINVESTAV",
            "english": "Center for Research and Advanced Studies",
            "ror_id": "https://ror.org/009eqmr18",
            "aliases": [
                "CENTRO DE INVESTIGACION Y DE ESTUDIOS AVANZADOS",
                "CINVESTAV", "CINVESTAV-IPN", "CINVESTAV DEL IPN",
                "CENTRO DE INVESTIGACION Y DE ESTUDIOS AVANZADOS DEL IPN",
            ],
        },
        "Universidad Autónoma Metropolitana": {
            "abbr": "UAM",
            "english": "Metropolitan Autonomous University",
            "ror_id": "https://ror.org/02kta5139",
            "aliases": [
                "UNIVERSIDAD AUTONOMA METROPOLITANA",
                "UAM", "U.A.M.",
                "UAM IZTAPALAPA", "UAM AZCAPOTZALCO", "UAM XOCHIMILCO",
                "UAM CUAJIMALPA", "UAM LERMA",
            ],
        },
        "Instituto Tecnológico y de Estudios Superiores de Monterrey": {
            "abbr": "ITESM",
            "english": "Monterrey Institute of Technology",
            "ror_id": "https://ror.org/03ayjn504",
            "aliases": [
                "TECNOLOGICO DE MONTERREY", "TEC DE MONTERREY",
                "ITESM", "INSTITUTO TECNOLOGICO DE MONTERREY",
                "INSTITUTO TECNOLOGICO Y DE ESTUDIOS SUPERIORES DE MONTERREY",
            ],
        },
        "El Colegio de México": {
            "abbr": "COLMEX",
            "english": "El Colegio de Mexico",
            "ror_id": "https://ror.org/03jne4y10",
            "aliases": [
                "EL COLEGIO DE MEXICO", "COLMEX",
                "COLEGIO DE MEXICO",
            ],
        },
        "Universidad de Guadalajara": {
            "abbr": "UDG",
            "english": "University of Guadalajara",
            "ror_id": "https://ror.org/043xj7k26",
            "aliases": [
                "UNIVERSIDAD DE GUADALAJARA", "UDG", "UDEG", "U DE G",
            ],
        },
        "Benemérita Universidad Autónoma de Puebla": {
            "abbr": "BUAP",
            "english": "Meritorious Autonomous University of Puebla",
            "ror_id": "https://ror.org/03p0pv290",
            "aliases": [
                "BENEMERITA UNIVERSIDAD AUTONOMA DE PUEBLA",
                "BUAP", "B.U.A.P.",
            ],
        },
        "Universidad Autónoma de Nuevo León": {
            "abbr": "UANL",
            "english": "Autonomous University of Nuevo Leon",
            "ror_id": "https://ror.org/01fh86n78",
            "aliases": [
                "UNIVERSIDAD AUTONOMA DE NUEVO LEON",
                "UANL", "U.A.N.L.",
            ],
        },
        "Universidad Autónoma del Estado de México": {
            "abbr": "UAEMEX",
            "english": "Autonomous University of the State of Mexico",
            "ror_id": "https://ror.org/00rz1yk83",
            "aliases": [
                "UNIVERSIDAD AUTONOMA DEL ESTADO DE MEXICO",
                "UAEMEX", "UAEM",
            ],
        },
        "Centro de Investigación Científica y de Educación Superior de Ensenada": {
            "abbr": "CICESE",
            "english": "CICESE",
            "ror_id": "https://ror.org/05h0rne59",
            "aliases": [
                "CICESE",
                "CENTRO DE INVESTIGACION CIENTIFICA Y DE EDUCACION SUPERIOR DE ENSENADA",
            ],
        },
        "Instituto Nacional de Astrofísica, Óptica y Electrónica": {
            "abbr": "INAOE",
            "english": "INAOE",
            "ror_id": "https://ror.org/01x5y2k90",
            "aliases": [
                "INAOE",
                "INSTITUTO NACIONAL DE ASTROFISICA OPTICA Y ELECTRONICA",
            ],
        },
        "Universidad Autónoma de San Luis Potosí": {
            "abbr": "UASLP",
            "english": "Autonomous University of San Luis Potosi",
            "ror_id": "https://ror.org/009g0sh51",
            "aliases": [
                "UNIVERSIDAD AUTONOMA DE SAN LUIS POTOSI",
                "UASLP",
            ],
        },
        "El Colegio de la Frontera Sur": {
            "abbr": "ECOSUR",
            "english": "El Colegio de la Frontera Sur",
            "ror_id": "https://ror.org/01r13mt82",
            "aliases": [
                "EL COLEGIO DE LA FRONTERA SUR", "ECOSUR",
            ],
        },
        "Centro de Investigación en Matemáticas": {
            "abbr": "CIMAT",
            "english": "Center for Research in Mathematics",
            "ror_id": "https://ror.org/03s82pf73",
            "aliases": [
                "CENTRO DE INVESTIGACION EN MATEMATICAS", "CIMAT",
            ],
        },
        "Universidad Veracruzana": {
            "abbr": "UV",
            "english": "University of Veracruz",
            "ror_id": "https://ror.org/05t1h8f27",
            "aliases": [
                "UNIVERSIDAD VERACRUZANA", "UV",
            ],
        },
        "Universidad Autónoma de Yucatán": {
            "abbr": "UADY",
            "english": "Autonomous University of Yucatan",
            "ror_id": "https://ror.org/02c3k0867",
            "aliases": [
                "UNIVERSIDAD AUTONOMA DE YUCATAN", "UADY",
            ],
        },
        "Centro de Investigaciones y Estudios Superiores en Antropología Social": {
            "abbr": "CIESAS",
            "english": "CIESAS",
            "ror_id": "https://ror.org/01kz1sq06",
            "aliases": [
                "CIESAS",
                "CENTRO DE INVESTIGACIONES Y ESTUDIOS SUPERIORES EN ANTROPOLOGIA SOCIAL",
            ],
        },
        "Instituto de Ecología": {
            "abbr": "INECOL",
            "english": "Institute of Ecology",
            "ror_id": "https://ror.org/02wmsc916",
            "aliases": [
                "INSTITUTO DE ECOLOGIA", "INECOL",
            ],
        },
        "Universidad Autónoma de Querétaro": {
            "abbr": "UAQ",
            "english": "Autonomous University of Queretaro",
            "ror_id": "https://ror.org/00bk4sb39",
            "aliases": [
                "UNIVERSIDAD AUTONOMA DE QUERETARO", "UAQ",
            ],
        },
    }

    def __init__(self) -> None:
        """Construye índice inverso de aliases para búsqueda rápida."""
        self._alias_index: dict[str, str] = {}
        for canonical, info in self.INSTITUTIONS.items():
            key = self._make_key(canonical)
            self._alias_index[key] = canonical
            for alias in info.get("aliases", []):
                self._alias_index[self._make_key(alias)] = canonical

    @staticmethod
    def _make_key(text: str) -> str:
        """Crea una clave normalizada para búsqueda."""
        text = text.strip().upper()
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in nfkd if not unicodedata.combining(c))
        text = re.sub(r"[^A-Z\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def normalize(self, institution: str) -> str:
        """Devuelve la forma canónica de una institución.

        Args:
            institution: Nombre de la institución (cualquier formato).

        Returns:
            Forma canónica o el input limpio si no hay match.

        Examples:
            >>> InstitutionNormalizer().normalize("UNAM")
            'Universidad Nacional Autónoma de México'
            >>> InstitutionNormalizer().normalize("TEC DE MONTERREY")
            'Instituto Tecnológico y de Estudios Superiores de Monterrey'
        """
        key = self._make_key(institution)
        if key in self._alias_index:
            return self._alias_index[key]

        # Fuzzy fallback: buscar coincidencia parcial
        best_match = self._fuzzy_lookup(key)
        if best_match:
            return best_match

        logger.debug(f"  Institución no reconocida: '{institution}'")
        return institution.strip().title()

    def _fuzzy_lookup(self, key: str) -> Optional[str]:
        """Busca la mejor coincidencia parcial en el índice."""
        try:
            from rapidfuzz import fuzz
            best_score = 0
            best_match = None
            for alias_key, canonical in self._alias_index.items():
                score = fuzz.ratio(key, alias_key)
                if score > best_score and score >= 80:
                    best_score = score
                    best_match = canonical
            return best_match
        except ImportError:
            return None

    def get_aliases(self, institution: str) -> list[str]:
        """Devuelve todos los aliases conocidos de una institución.

        Args:
            institution: Nombre de la institución.

        Returns:
            Lista con forma canónica, abreviatura, nombre en inglés,
            y aliases adicionales.
        """
        canonical = self.normalize(institution)
        info = self.INSTITUTIONS.get(canonical)
        if not info:
            return [institution.strip()]

        result = [canonical]
        if info.get("abbr"):
            result.append(info["abbr"])
        if info.get("english"):
            result.append(info["english"])
        result.extend(info.get("aliases", []))

        # Deduplicar preservando orden
        seen: set[str] = set()
        unique: list[str] = []
        for a in result:
            if a.lower() not in seen:
                seen.add(a.lower())
                unique.append(a)
        return unique

    def get_ror_id(self, institution: str) -> Optional[str]:
        """Devuelve el ROR ID de una institución si es conocido.

        Args:
            institution: Nombre de la institución.

        Returns:
            ROR ID (URL) o None.

        Examples:
            >>> InstitutionNormalizer().get_ror_id("UNAM")
            'https://ror.org/01tmp8f25'
        """
        canonical = self.normalize(institution)
        info = self.INSTITUTIONS.get(canonical)
        return info.get("ror_id") if info else None

    def get_abbreviation(self, institution: str) -> Optional[str]:
        """Devuelve la abreviatura de la institución."""
        canonical = self.normalize(institution)
        info = self.INSTITUTIONS.get(canonical)
        return info.get("abbr") if info else None

    def normalize_record_institution(self, institution: str) -> dict:
        """Normaliza la institución de un registro SNII.

        Returns:
            Dict con canonical, aliases, ror_id, abbreviation.
        """
        canonical = self.normalize(institution)
        return {
            "normalized_institution": canonical,
            "institution_aliases": self.get_aliases(institution),
            "ror_id": self.get_ror_id(institution),
            "abbreviation": self.get_abbreviation(institution),
        }
