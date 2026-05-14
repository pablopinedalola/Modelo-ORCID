#!/usr/bin/env python3
"""
ingest_snii.py — Etapa 1 del pipeline de ingestión de datos SNII.

Script CLI que ejecuta el pipeline completo de ingestión:
    1. Lectura del archivo Excel/CSV del padrón SNII
    2. Validación de columnas esperadas
    3. Limpieza básica (whitespace, NaN, duplicados)
    4. Normalización simple (uppercase, acentos)
    5. Generación de perfiles básicos con searchable_text
    6. Exportación a JSON y CSV procesados

Uso:
    python scripts/ingest_snii.py data/raw/SNII_2024.xlsx
    python scripts/ingest_snii.py data/raw/padron.csv --output-dir data/processed
    python scripts/ingest_snii.py data/raw/SNII_2024.xlsx --stats
    python scripts/ingest_snii.py data/raw/SNII_2024.xlsx --dry-run

Salida:
    data/processed/snii_profiles.json    — Perfiles completos
    data/processed/snii_profiles.csv     — Tabla tabular para inspección
    data/processed/snii_stats.json       — Estadísticas de ingestión
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ajustar PYTHONPATH para importaciones del proyecto ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.loader.snii_loader import SNIILoader
from src.profiles.profile_builder import ProfileBuilder

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_snii")


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    show_stats: bool = False,
) -> dict:
    """Ejecuta el pipeline completo de ingestión SNII.

    Args:
        input_path: Ruta al archivo Excel o CSV del padrón SNII.
        output_dir: Directorio de salida para archivos procesados.
        dry_run: Si True, solo valida sin escribir archivos.
        show_stats: Si True, muestra estadísticas detalladas.

    Returns:
        Diccionario con estadísticas del pipeline.

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe.
        ValueError: Si el archivo tiene formato no soportado.
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("🚀 PIPELINE DE INGESTIÓN SNII — ETAPA 1")
    logger.info("=" * 60)
    logger.info(f"   Entrada:  {input_path}")
    logger.info(f"   Salida:   {output_dir}")
    logger.info(f"   Dry-run:  {dry_run}")
    logger.info("")

    # ── Paso 1: Cargar y limpiar con SNIILoader ─────────────────────────
    logger.info("📥 PASO 1/4: Carga y limpieza del archivo SNII...")
    loader = SNIILoader()
    records = loader.load(input_path)

    if not records:
        logger.error("❌ No se cargaron registros. Verifica el archivo de entrada.")
        sys.exit(1)

    logger.info(f"   ✅ {len(records)} registros cargados y limpios")
    logger.info("")

    # ── Paso 2: Validación de columnas ──────────────────────────────────
    logger.info("🔍 PASO 2/4: Validación de columnas...")
    validation = _validate_records(records)
    _print_validation(validation)
    logger.info("")

    # ── Paso 3: Generar perfiles básicos ────────────────────────────────
    logger.info("👤 PASO 3/4: Generación de perfiles básicos...")
    builder = ProfileBuilder()
    profiles = builder.build_profiles(records)
    logger.info(f"   ✅ {len(profiles)} perfiles generados")

    # Muestra un ejemplo
    if profiles:
        sample = profiles[0]
        logger.info(f"   📋 Ejemplo de perfil:")
        logger.info(f"      nombre:          {sample['nombre_completo']}")
        logger.info(f"      institución:     {sample['institucion']}")
        logger.info(f"      área:            {sample['area']}")
        logger.info(f"      disciplina:      {sample['disciplina']}")
        logger.info(f"      searchable_text: {sample['searchable_text'][:80]}...")
    logger.info("")

    # ── Paso 4: Exportación ─────────────────────────────────────────────
    logger.info("💾 PASO 4/4: Exportación de datos procesados...")

    if dry_run:
        logger.info("   ⏭️  Dry-run: se omite la escritura de archivos.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _export_json(profiles, output_dir / "snii_profiles.json")
        _export_csv(profiles, output_dir / "snii_profiles.csv")
        logger.info(f"   ✅ Archivos exportados en {output_dir}")
    logger.info("")

    # ── Estadísticas ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    stats = _build_stats(records, profiles, loader, elapsed)

    if show_stats or not dry_run:
        _print_stats(stats)

    if not dry_run:
        stats_path = output_dir / "snii_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"   📊 Estadísticas guardadas en {stats_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ PIPELINE COMPLETADO en {elapsed:.2f}s")
    logger.info("=" * 60)

    return stats


# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═════════════════════════════════════════════════════════════════════════════

def _validate_records(records: list) -> dict:
    """Valida la calidad de los registros cargados.

    Args:
        records: Lista de SNIIRecord.

    Returns:
        Diccionario con métricas de validación.
    """
    total = len(records)
    missing_nombre = sum(1 for r in records if not r.nombre)
    missing_paterno = sum(1 for r in records if not r.paterno)
    missing_institucion = sum(1 for r in records if not r.institucion)
    missing_area = sum(1 for r in records if not r.area)
    missing_disciplina = sum(1 for r in records if not r.disciplina)
    missing_nivel = sum(1 for r in records if r.nivel.value == "?")

    return {
        "total": total,
        "campos": {
            "nombre": {"presentes": total - missing_nombre, "faltantes": missing_nombre},
            "paterno": {"presentes": total - missing_paterno, "faltantes": missing_paterno},
            "institucion": {"presentes": total - missing_institucion, "faltantes": missing_institucion},
            "area": {"presentes": total - missing_area, "faltantes": missing_area},
            "disciplina": {"presentes": total - missing_disciplina, "faltantes": missing_disciplina},
            "nivel": {"presentes": total - missing_nivel, "faltantes": missing_nivel},
        },
    }


def _print_validation(validation: dict) -> None:
    """Imprime el reporte de validación al logger."""
    total = validation["total"]
    for campo, info in validation["campos"].items():
        pct = (info["presentes"] / total * 100) if total > 0 else 0
        status = "✅" if info["faltantes"] == 0 else "⚠️"
        logger.info(
            f"   {status} {campo:15s}  "
            f"{info['presentes']:>6d}/{total} ({pct:5.1f}%)  "
            f"{'— ' + str(info['faltantes']) + ' faltantes' if info['faltantes'] else ''}"
        )


def _build_stats(records, profiles, loader, elapsed) -> dict:
    """Genera estadísticas completas del pipeline."""
    from collections import Counter

    niveles = Counter(r.nivel.value for r in records)
    areas = Counter(r.area for r in records if r.area)
    top_inst = Counter(r.institucion for r in records if r.institucion)

    return {
        "timestamp": datetime.now().isoformat(),
        "archivo_fuente": str(getattr(loader, '_last_filepath', 'N/A')),
        "pipeline": {
            "total_registros_cargados": len(records),
            "total_perfiles_generados": len(profiles),
            "tiempo_ejecucion_segundos": round(elapsed, 2),
        },
        "distribucion_nivel": dict(niveles.most_common()),
        "distribucion_area": dict(areas.most_common()),
        "top_10_instituciones": dict(top_inst.most_common(10)),
        "columnas_mapeadas": loader.column_map,
    }


def _print_stats(stats: dict) -> None:
    """Imprime estadísticas al logger."""
    logger.info("📊 ESTADÍSTICAS DE INGESTIÓN:")
    logger.info(f"   Registros: {stats['pipeline']['total_registros_cargados']}")
    logger.info(f"   Perfiles:  {stats['pipeline']['total_perfiles_generados']}")
    logger.info(f"   Tiempo:    {stats['pipeline']['tiempo_ejecucion_segundos']}s")

    logger.info("   Por nivel SNI:")
    for nivel, count in stats["distribucion_nivel"].items():
        logger.info(f"      {nivel}: {count}")

    logger.info("   Por área:")
    for area, count in list(stats["distribucion_area"].items())[:7]:
        logger.info(f"      {area}: {count}")

    logger.info("   Top instituciones:")
    for inst, count in list(stats["top_10_instituciones"].items())[:5]:
        logger.info(f"      {inst[:50]}: {count}")


def _export_json(profiles: list[dict], path: Path) -> None:
    """Exporta perfiles a JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    logger.info(f"   📄 JSON: {path} ({len(profiles)} perfiles)")


def _export_csv(profiles: list[dict], path: Path) -> None:
    """Exporta perfiles a CSV (sin searchable_text para legibilidad)."""
    import pandas as pd

    df = pd.DataFrame(profiles)
    # Truncar searchable_text para legibilidad en CSV
    if "searchable_text" in df.columns:
        df["searchable_text"] = df["searchable_text"].str[:100]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"   📄 CSV:  {path} ({len(df)} filas)")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point del CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de ingestión del padrón SNII — Etapa 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/ingest_snii.py data/raw/SNII_2024.xlsx
  python scripts/ingest_snii.py data/raw/padron.csv --dry-run
  python scripts/ingest_snii.py data/raw/SNII_2024.xlsx --stats --output-dir data/processed
        """,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Ruta al archivo Excel (.xlsx) o CSV (.csv) del padrón SNII",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Directorio de salida (default: {PROCESSED_DATA_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validar sin escribir archivos de salida",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estadísticas detalladas al final",
    )

    args = parser.parse_args()

    # Validar archivo de entrada
    if not args.input_file.exists():
        logger.error(f"❌ Archivo no encontrado: {args.input_file}")
        logger.info(f"   Coloca el archivo del padrón SNII en {RAW_DATA_DIR}/")
        sys.exit(1)

    if args.input_file.suffix.lower() not in (".xlsx", ".xls", ".csv"):
        logger.error(f"❌ Formato no soportado: {args.input_file.suffix}")
        logger.info("   Formatos aceptados: .xlsx, .xls, .csv")
        sys.exit(1)

    try:
        run_pipeline(
            input_path=args.input_file,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            show_stats=args.stats,
        )
    except Exception as e:
        logger.error(f"❌ Error fatal en el pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
