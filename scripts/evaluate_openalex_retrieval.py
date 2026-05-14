#!/usr/bin/env python3
"""
evaluate_openalex_retrieval.py — Evalúa cobertura y volumen del dataset descargado.
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "openalex"
AUTHORS_DIR = DATA_DIR / "authors"
WORKS_DIR = DATA_DIR / "works"


def main():
    print("=" * 60)
    print("  EVALUATE OPENALEX RETRIEVAL")
    print("=" * 60)

    # Contar archivos reales
    author_files = list(AUTHORS_DIR.glob("*.json"))
    works_files = list(WORKS_DIR.glob("*_works.json"))

    total_papers = 0
    all_topics = set()
    all_institutions = set()
    all_dois = set()
    papers_with_abstract = 0
    papers_with_refs = 0

    for wf in works_files:
        with open(wf) as f:
            papers = json.load(f)
        total_papers += len(papers)
        for p in papers:
            # Topics
            for t in p.get("topics", []):
                n = t.get("name")
                if n:
                    all_topics.add(n)
            # Institutions
            for inst in p.get("institutions", []):
                if inst:
                    all_institutions.add(inst)
            # DOIs
            doi = p.get("doi")
            if doi:
                all_dois.add(doi)
            # Abstract coverage
            if p.get("abstract"):
                papers_with_abstract += 1
            # Refs
            if p.get("referenced_works"):
                papers_with_refs += 1

    print(f"\n👤 Total autores:       {len(author_files)}")
    print(f"📄 Total papers:        {total_papers}")
    print(f"🏷️  Total topics únicos: {len(all_topics)}")
    print(f"🏫 Total instituciones: {len(all_institutions)}")
    print(f"🔗 Total DOIs:          {len(all_dois)}")
    print(f"\n📊 Cobertura:")
    if total_papers > 0:
        print(f"   Abstracts:    {papers_with_abstract}/{total_papers} ({100*papers_with_abstract/total_papers:.0f}%)")
        print(f"   References:   {papers_with_refs}/{total_papers} ({100*papers_with_refs/total_papers:.0f}%)")
    else:
        print("   No hay papers.")

    # Tamaño en disco
    total_bytes = sum(f.stat().st_size for f in AUTHORS_DIR.glob("*.json"))
    total_bytes += sum(f.stat().st_size for f in WORKS_DIR.glob("*.json"))
    print(f"\n💾 Tamaño total en disco: {total_bytes / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("  EVALUATION DONE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
