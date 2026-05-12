"""
snii_loader.py — Carga, validación y limpieza del padrón SNII.

Responsabilidades:
    - Leer CSV o Excel del padrón SNII (detección automática de formato)
    - Detectar encoding (UTF-8, Latin-1, cp1252)
    - Normalizar nombres de columnas
    - Limpiar valores: strip whitespace, manejar NaN, estandarizar
    - Convertir filas a dataclasses tipadas (SNIIRecord)

Examples:
    >>> loader = SNIILoader()
    >>> records = loader.load("data/raw/snii_sample.csv")
    >>> print(f"Cargados {len(records)} investigadores")
    Cargados 20 investigadores
    >>> print(records[0].full_name)
    CARLOS ALBERTO GARCIA LOPEZ
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.schemas import SNIIRecord, SNIILevel
from config import SNII_EXPECTED_COLUMNS

logger = logging.getLogger(__name__)


class SNIILoader:
    """Carga y procesa archivos del padrón SNII.

    Soporta CSV (.csv) y Excel (.xlsx, .xls) con detección automática
    de formato y encoding. Mapea columnas flexiblemente usando el
    diccionario de aliases definido en config.SNII_EXPECTED_COLUMNS.

    Attributes:
        column_map: Mapeo de nombre canónico → nombre real en el archivo.
        raw_df: DataFrame crudo tras carga (antes de limpieza).

    Examples:
        >>> loader = SNIILoader()
        >>> records = loader.load("data/raw/padron_snii_2024.xlsx")
        >>> len(records)
        35000
    """

    # Encodings a probar en orden de prioridad
    ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

    def __init__(self) -> None:
        self.column_map: dict[str, str] = {}
        self.raw_df: Optional[pd.DataFrame] = None

    def load(self, filepath: str | Path) -> list[SNIIRecord]:
        """Pipeline completo: leer → limpiar → convertir a records.

        Args:
            filepath: Ruta al archivo CSV o Excel del padrón.

        Returns:
            Lista de SNIIRecord con datos limpios y tipados.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el formato no es soportado o faltan columnas críticas.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        logger.info(f"📂 Cargando padrón SNII desde: {filepath}")

        # Paso 1: Leer archivo
        df = self._read_file(filepath)
        self.raw_df = df.copy()
        logger.info(f"   Filas crudas: {len(df)}")

        # Paso 2: Mapear columnas
        df = self._map_columns(df)

        # Paso 3: Limpiar datos
        df = self._clean(df)
        logger.info(f"   Filas tras limpieza: {len(df)}")

        # Paso 4: Convertir a records
        records = self._to_records(df)
        logger.info(f"✅ {len(records)} investigadores cargados exitosamente")

        return records

    def _read_file(self, filepath: Path) -> pd.DataFrame:
        """Lee un archivo CSV o Excel con detección automática de encoding.

        Args:
            filepath: Ruta al archivo.

        Returns:
            DataFrame con los datos crudos.

        Raises:
            ValueError: Si el formato del archivo no es soportado.
        """
        suffix = filepath.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            logger.info("   Formato detectado: Excel")
            return pd.read_excel(filepath, engine="openpyxl")

        elif suffix == ".csv":
            logger.info("   Formato detectado: CSV")
            return self._read_csv_with_encoding(filepath)

        else:
            raise ValueError(
                f"Formato no soportado: '{suffix}'. "
                f"Use .csv, .xlsx o .xls"
            )

    def _read_csv_with_encoding(self, filepath: Path) -> pd.DataFrame:
        """Intenta leer un CSV probando múltiples encodings.

        Args:
            filepath: Ruta al archivo CSV.

        Returns:
            DataFrame leído con el encoding correcto.

        Raises:
            ValueError: Si ningún encoding funciona.
        """
        errors_found: list[str] = []

        for encoding in self.ENCODINGS:
            try:
                df = pd.read_csv(filepath, encoding=encoding, dtype=str)
                logger.info(f"   Encoding detectado: {encoding}")
                return df
            except (UnicodeDecodeError, UnicodeError) as e:
                errors_found.append(f"{encoding}: {e}")
                continue

        raise ValueError(
            f"No se pudo leer el archivo con ningún encoding. "
            f"Intentados: {', '.join(self.ENCODINGS)}"
        )

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mapea columnas del archivo a nombres canónicos.

        Busca cada columna esperada entre los aliases definidos en
        SNII_EXPECTED_COLUMNS. Las columnas se normalizan a lowercase
        y sin acentos para el matching.

        Args:
            df: DataFrame con columnas originales.

        Returns:
            DataFrame con columnas renombradas a nombres canónicos.
        """
        # Normalizar nombres de columnas existentes
        original_columns = df.columns.tolist()
        normalized_cols = {
            self._normalize_col_name(col): col for col in original_columns
        }

        rename_map = {}
        for canonical_name, aliases in SNII_EXPECTED_COLUMNS.items():
            # Buscar el nombre canónico directamente
            all_options = [canonical_name] + aliases
            for alias in all_options:
                norm_alias = self._normalize_col_name(alias)
                if norm_alias in normalized_cols:
                    rename_map[normalized_cols[norm_alias]] = canonical_name
                    self.column_map[canonical_name] = normalized_cols[norm_alias]
                    break

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(
                f"   Columnas mapeadas: {list(rename_map.values())}"
            )

        # Verificar columnas críticas
        critical = {"nombre", "paterno"}
        found = set(df.columns) & critical
        if not critical.issubset(found):
            missing = critical - found
            logger.warning(
                f"⚠️  Columnas críticas no encontradas: {missing}. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        return df

    @staticmethod
    def _normalize_col_name(name: str) -> str:
        """Normaliza un nombre de columna para comparación.

        Args:
            name: Nombre original de la columna.

        Returns:
            Nombre en lowercase, sin espacios, sin acentos.
        """
        import unicodedata
        name = name.strip().lower()
        name = name.replace(" ", "_").replace("-", "_")
        # Remover acentos
        nfkd = unicodedata.normalize("NFKD", name)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y estandariza los datos del DataFrame.

        Operaciones:
            1. Eliminar filas completamente vacías
            2. Strip whitespace en todas las columnas string
            3. Reemplazar NaN/None con string vacío
            4. Convertir a uppercase para consistencia
            5. Eliminar duplicados exactos

        Args:
            df: DataFrame con datos mapeados.

        Returns:
            DataFrame limpio.
        """
        initial_rows = len(df)

        # Eliminar filas vacías
        df = df.dropna(how="all")

        # Strip whitespace y reemplazar NaN
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

        # Eliminar duplicados si existe la combinación nombre+paterno+institucion
        dedup_cols = [c for c in ["nombre", "paterno", "institucion"] if c in df.columns]
        if dedup_cols:
            before = len(df)
            df = df.drop_duplicates(subset=dedup_cols, keep="first")
            removed = before - len(df)
            if removed > 0:
                logger.info(f"   Duplicados eliminados: {removed}")

        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.info(f"   Filas eliminadas (vacías/duplicados): {dropped}")

        return df.reset_index(drop=True)

    def _to_records(self, df: pd.DataFrame) -> list[SNIIRecord]:
        """Convierte cada fila del DataFrame a un SNIIRecord tipado.

        Args:
            df: DataFrame limpio con columnas canónicas.

        Returns:
            Lista de SNIIRecord.
        """
        records = []
        for _, row in df.iterrows():
            try:
                record = SNIIRecord(
                    nombre=self._get_field(row, "nombre"),
                    paterno=self._get_field(row, "paterno"),
                    materno=self._get_field(row, "materno"),
                    institucion=self._get_field(row, "institucion"),
                    dependencia=self._get_field(row, "dependencia"),
                    subdependencia=self._get_field(row, "subdependencia"),
                    area=self._get_field(row, "area"),
                    disciplina=self._get_field(row, "disciplina"),
                    nivel=SNIILevel.from_string(self._get_field(row, "nivel")),
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"⚠️  Error procesando fila: {e}")
                continue

        return records

    @staticmethod
    def _get_field(row: pd.Series, field: str) -> str:
        """Extrae un campo de una fila de forma segura.

        Args:
            row: Fila del DataFrame.
            field: Nombre del campo a extraer.

        Returns:
            Valor del campo como string, o vacío si no existe.
        """
        if field in row.index:
            val = row[field]
            if pd.isna(val):
                return ""
            return str(val).strip()
        return ""

    def summary(self, records: list[SNIIRecord]) -> dict:
        """Genera estadísticas resumen de los registros cargados.

        Args:
            records: Lista de registros SNII.

        Returns:
            Diccionario con conteos por nivel, área e institución.

        Examples:
            >>> summary = loader.summary(records)
            >>> print(summary["total"])
            20
        """
        from collections import Counter

        niveles = Counter(r.nivel.value for r in records)
        areas = Counter(r.area for r in records if r.area)
        instituciones = Counter(r.institucion for r in records if r.institucion)

        return {
            "total": len(records),
            "por_nivel": dict(niveles.most_common()),
            "por_area": dict(areas.most_common(10)),
            "top_instituciones": dict(instituciones.most_common(10)),
        }
