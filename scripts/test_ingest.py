#!/usr/bin/env python3
"""
test_ingest.py — Verifica el pipeline de ingestión con datos de prueba.

Crea un Excel de muestra y ejecuta el pipeline completo para validar
que todo funcione correctamente sin depender de un archivo real.

Uso:
    python scripts/test_ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ajustar PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def create_sample_excel() -> Path:
    """Crea un Excel SNII de prueba con datos ficticios."""

    data = {
        "NOMBRE": [
            "CARLOS ALBERTO",
            "MARÍA FERNANDA",
            "JUAN PABLO",
            "ANA LUCÍA",
            "ROBERTO",
            "DIANA PATRICIA",
            "ALEJANDRO",
            "GABRIELA",
            "FRANCISCO JAVIER",
            "PATRICIA ELENA",
            "LUIS ENRIQUE",
            "CLAUDIA MARCELA",
            "MIGUEL ÁNGEL",
            "VERÓNICA",
            "JORGE ANTONIO",
        ],
        "PATERNO": [
            "GARCÍA",
            "HERNÁNDEZ",
            "LÓPEZ",
            "MARTÍNEZ",
            "RODRÍGUEZ",
            "SÁNCHEZ",
            "RAMÍREZ",
            "TORRES",
            "FLORES",
            "GÓMEZ",
            "DÍAZ",
            "VÁZQUEZ",
            "RUIZ",
            "MORALES",
            "ORTIZ",
        ],
        "MATERNO": [
            "LÓPEZ",
            "SÁNCHEZ",
            "MARTÍNEZ",
            "GONZÁLEZ",
            "PÉREZ",
            "RAMÍREZ",
            "CRUZ",
            "MORALES",
            "VARGAS",
            "CASTILLO",
            "HERRERA",
            "MEDINA",
            "JIMÉNEZ",
            "SILVA",
            "AGUIRRE",
        ],
        "INSTITUCIÓN": [
            "UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO",
            "INSTITUTO POLITÉCNICO NACIONAL",
            "CENTRO DE INVESTIGACIÓN Y DE ESTUDIOS AVANZADOS DEL IPN",
            "UNIVERSIDAD AUTÓNOMA METROPOLITANA",
            "UNIVERSIDAD DE GUADALAJARA",
            "BENEMÉRITA UNIVERSIDAD AUTÓNOMA DE PUEBLA",
            "UNIVERSIDAD AUTÓNOMA DE NUEVO LEÓN",
            "UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MORELOS",
            "UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO",
            "INSTITUTO POLITÉCNICO NACIONAL",
            "CENTRO DE INVESTIGACIÓN CIENTÍFICA Y DE EDUCACIÓN SUPERIOR DE ENSENADA",
            "UNIVERSIDAD AUTÓNOMA DE SAN LUIS POTOSÍ",
            "UNIVERSIDAD VERACRUZANA",
            "EL COLEGIO DE MÉXICO",
            "UNIVERSIDAD AUTÓNOMA DE QUERÉTARO",
        ],
        "DEPENDENCIA": [
            "INSTITUTO DE FÍSICA",
            "ESCUELA SUPERIOR DE INGENIERÍA MECÁNICA Y ELÉCTRICA",
            "DEPARTAMENTO DE MATEMÁTICAS",
            "CIENCIAS BÁSICAS E INGENIERÍA",
            "CENTRO UNIVERSITARIO DE CIENCIAS EXACTAS",
            "FACULTAD DE CIENCIAS FÍSICO MATEMÁTICAS",
            "FACULTAD DE INGENIERÍA MECÁNICA Y ELÉCTRICA",
            "CENTRO DE INVESTIGACIÓN EN INGENIERÍA Y CIENCIAS APLICADAS",
            "INSTITUTO DE INVESTIGACIONES EN MATERIALES",
            "ESCUELA NACIONAL DE CIENCIAS BIOLÓGICAS",
            "DEPARTAMENTO DE ÓPTICA",
            "FACULTAD DE CIENCIAS",
            "INSTITUTO DE INVESTIGACIONES BIOLÓGICAS",
            "CENTRO DE ESTUDIOS INTERNACIONALES",
            "FACULTAD DE QUÍMICA",
        ],
        "SUBDEPENDENCIA": [
            "DEPARTAMENTO DE PARTÍCULAS ELEMENTALES",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "LABORATORIO DE MATERIALES AVANZADOS",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "ÁREA": [
            "I",
            "VII",
            "I",
            "I",
            "II",
            "I",
            "VII",
            "VII",
            "I",
            "II",
            "I",
            "I",
            "II",
            "V",
            "II",
        ],
        "DISCIPLINA": [
            "FÍSICA DE PARTÍCULAS",
            "INGENIERÍA ELÉCTRICA",
            "MATEMÁTICAS APLICADAS",
            "CIENCIAS DE LA COMPUTACIÓN",
            "BIOQUÍMICA",
            "FÍSICA TEÓRICA",
            "INGENIERÍA MECATRÓNICA",
            "NANOTECNOLOGÍA",
            "CIENCIA DE MATERIALES",
            "MICROBIOLOGÍA",
            "ÓPTICA CUÁNTICA",
            "ASTROFÍSICA",
            "ECOLOGÍA",
            "RELACIONES INTERNACIONALES",
            "QUÍMICA ORGÁNICA",
        ],
        "NIVEL": [
            "SNI II",
            "Candidato",
            "SNI III",
            "SNI I",
            "SNI I",
            "SNI II",
            "Candidato",
            "SNI I",
            "Emérito",
            "SNI II",
            "SNI III",
            "SNI I",
            "SNI II",
            "SNI I",
            "Candidato",
        ],
    }

    df = pd.DataFrame(data)
    filepath = RAW_DATA_DIR / "snii_sample.xlsx"
    df.to_excel(filepath, index=False, engine="openpyxl")
    print(f"✅ Archivo de prueba creado: {filepath}")
    print(f"   {len(df)} registros, {len(df.columns)} columnas")
    return filepath


def main():
    """Ejecuta el test de ingestión."""
    print("=" * 60)
    print("🧪 TEST DE INGESTIÓN SNII — Datos de prueba")
    print("=" * 60)
    print()

    # 1. Crear archivo de muestra
    print("📝 Creando archivo Excel de muestra...")
    sample_path = create_sample_excel()
    print()

    # 2. Ejecutar pipeline
    from scripts.ingest_snii import run_pipeline

    stats = run_pipeline(
        input_path=sample_path,
        output_dir=PROCESSED_DATA_DIR,
        dry_run=False,
        show_stats=True,
    )

    # 3. Verificar outputs
    print()
    print("🔎 VERIFICACIÓN DE SALIDAS:")

    json_path = PROCESSED_DATA_DIR / "snii_profiles.json"
    csv_path = PROCESSED_DATA_DIR / "snii_profiles.csv"
    stats_path = PROCESSED_DATA_DIR / "snii_stats.json"

    for path in [json_path, csv_path, stats_path]:
        exists = "✅" if path.exists() else "❌"
        size = f"({path.stat().st_size:,} bytes)" if path.exists() else "(no existe)"
        print(f"   {exists} {path.name} {size}")

    # 4. Verificar contenido JSON
    if json_path.exists():
        import json
        with open(json_path) as f:
            profiles = json.load(f)
        print(f"\n📋 Verificación de contenido ({len(profiles)} perfiles):")

        for i, p in enumerate(profiles[:3]):
            print(f"\n   --- Perfil {i+1} ---")
            print(f"   nombre_completo:  {p['nombre_completo']}")
            print(f"   institucion:      {p['institucion']}")
            print(f"   area:             {p['area']} ({p['area_nombre']})")
            print(f"   disciplina:       {p['disciplina']}")
            print(f"   nivel:            {p['nivel']} ({p['nivel_nombre']})")
            print(f"   searchable_text:  {p['searchable_text'][:80]}...")

        # Verificar que searchable_text no tiene acentos
        sample_text = profiles[0]["searchable_text"]
        has_accents = any(
            ord(c) > 127 for c in sample_text
        )
        accent_check = "❌ TIENE acentos" if has_accents else "✅ Sin acentos (normalizado)"
        print(f"\n   Verificación de normalización: {accent_check}")

    print()
    print("=" * 60)
    print("✅ TEST COMPLETADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()
