"""
config.py — Configuración global del sistema Modelo-ORCID.

Centraliza parámetros de API, modelos, rutas y pesos de evidencia.
Los valores sensibles se leen de variables de entorno con fallback a defaults.
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# RUTAS DEL PROYECTO
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

# Crear directorios si no existen
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CREDENCIALES DE APIs
# ═══════════════════════════════════════════════════════════════════════════════

# ORCID Public API — Obtener en https://orcid.org/developer-tools
ORCID_CLIENT_ID = os.getenv("ORCID_CLIENT_ID", "")
ORCID_CLIENT_SECRET = os.getenv("ORCID_CLIENT_SECRET", "")
ORCID_API_BASE = "https://pub.orcid.org/v3.0"

# OpenAlex — API key gratuito en https://openalex.org/settings/api
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "modelo-orcid@example.com")
OPENALEX_API_BASE = "https://api.openalex.org"

# ROR — No requiere autenticación
ROR_API_BASE = "https://api.ror.org/v2/organizations"

# ═══════════════════════════════════════════════════════════════════════════════
# MODELO DE EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL REFINEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

MAX_ITERATIONS = 10
CONVERGENCE_EPSILON = 0.01
CONFIDENCE_THRESHOLD = 0.85

# Pesos para las dimensiones de evidencia (deben sumar 1.0)
EVIDENCE_WEIGHTS = {
    "name": 0.25,
    "institution": 0.20,
    "area": 0.10,
    "publication": 0.20,
    "coauthor": 0.10,
    "temporal": 0.05,
    "semantic": 0.10,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE BÚSQUEDA Y RETRIEVAL HÍBRIDO
# ═══════════════════════════════════════════════════════════════════════════════

# Pesos para la búsqueda híbrida (dense vs sparse)
HYBRID_SEMANTIC_WEIGHT = 0.65
HYBRID_LEXICAL_WEIGHT = 0.35

# Máximo de candidatos a recuperar por fuente
MAX_CANDIDATES_PER_SOURCE = 10

# Umbral mínimo de similitud fuzzy para considerar un candidato
MIN_FUZZY_THRESHOLD = 60

# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════

ORCID_RATE_LIMIT = 24          # requests/segundo (Public API)
OPENALEX_RATE_LIMIT = 10       # requests/segundo (polite pool)
ROR_RATE_LIMIT = 50            # requests/segundo

# Retry con backoff exponencial
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMNAS ESPERADAS DEL PADRÓN SNII
# ═══════════════════════════════════════════════════════════════════════════════

SNII_EXPECTED_COLUMNS = {
    "nombre": ["nombre", "nombres", "given_name", "first_name"],
    "paterno": ["paterno", "apellido_paterno", "apellido1", "last_name"],
    "materno": ["materno", "apellido_materno", "apellido2", "second_last_name"],
    "institucion": ["institucion", "institución", "institution", "inst"],
    "dependencia": ["dependencia", "dependency", "department"],
    "subdependencia": ["subdependencia", "sub_dependencia", "subdepartment"],
    "area": ["area", "área", "area_conocimiento"],
    "disciplina": ["disciplina", "discipline", "field"],
    "nivel": ["nivel", "level", "snii_level", "nivel_snii"],
}
