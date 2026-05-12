"""
run_server.py -- Levanta el Academic Explorer.

Usage:
    python run_server.py                   # Solo servidor (datos ya generados)
    python run_server.py --run-pipeline    # Ejecutar pipeline + servidor
    python run_server.py --port 8080       # Puerto personalizado
"""

import argparse
import logging
import subprocess
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Modelo-ORCID Academic Explorer"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Puerto del servidor (default: 8000)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--run-pipeline", action="store_true",
        help="Ejecutar pipeline antes de levantar servidor",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Investigadores a procesar en pipeline (default: 5)",
    )
    parser.add_argument(
        "--skip-semantic", action="store_true",
        help="Saltar semantic matching en pipeline",
    )

    args = parser.parse_args()

    # Optionally run pipeline first
    if args.run_pipeline:
        print("=" * 60)
        print("  Ejecutando pipeline...")
        print("=" * 60)
        cmd = [
            sys.executable, "main.py",
            "--limit", str(args.limit),
            "--save",
        ]
        if args.skip_semantic:
            cmd.append("--skip-semantic")

        result = subprocess.run(cmd, cwd=".")
        if result.returncode != 0:
            print("Error en pipeline. Abortando.")
            sys.exit(1)

    print()
    print("=" * 60)
    print(f"  Modelo-ORCID Academic Explorer")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 60)
    print()

    uvicorn.run(
        "api.routes:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
