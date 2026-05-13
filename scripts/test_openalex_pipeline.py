#!/usr/bin/env python3
"""
test_openalex_pipeline.py — Carga JSONs descargados e imprime estadísticas reales.
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "openalex"
AUTHORS_DIR = DATA_DIR / "authors"
WORKS_DIR = DATA_DIR / "works"


def load_all_works() -> list[dict]:
    papers = []
    for f in sorted(WORKS_DIR.glob("*_works.json")):
        with open(f) as fh:
            papers.extend(json.load(fh))
    return papers


def main():
    print("=" * 60)
    print("  TEST OPENALEX PIPELINE")
    print("=" * 60)

    # --- Autores ---
    author_files = sorted(AUTHORS_DIR.glob("*.json"))
    print(f"\n👤 Autores descargados: {len(author_files)}")
    for af in author_files:
        with open(af) as f:
            a = json.load(f)
        print(f"   - {a.get('display_name', '?')}  |  works={a.get('works_count',0)}  cited={a.get('cited_by_count',0)}  h={a.get('summary_stats',{}).get('h_index','?')}")

    # --- Papers ---
    papers = load_all_works()
    print(f"\n📄 Total papers descargados: {len(papers)}")

    # Top cited
    by_cites = sorted(papers, key=lambda p: p.get("cited_by_count", 0), reverse=True)
    print("\n🏆 Top 5 papers más citados:")
    for p in by_cites[:5]:
        t = (p.get("title") or "")[:70]
        print(f"   [{p.get('cited_by_count',0):>5} citas] {t}")

    # Top topics
    topic_counter = Counter()
    for p in papers:
        for t in p.get("topics", []):
            name = t.get("name")
            if name:
                topic_counter[name] += 1
    print(f"\n🏷️  Topics únicos: {len(topic_counter)}")
    print("   Top 5:")
    for name, count in topic_counter.most_common(5):
        print(f"   - {name}: {count} papers")

    # Top institutions
    inst_counter = Counter()
    for p in papers:
        for inst in p.get("institutions", []):
            if inst:
                inst_counter[inst] += 1
    print(f"\n🏫 Instituciones únicas: {len(inst_counter)}")
    print("   Top 5:")
    for name, count in inst_counter.most_common(5):
        print(f"   - {name}: {count} papers")

    # Años
    year_counter = Counter()
    for p in papers:
        y = p.get("publication_year")
        if y:
            year_counter[y] += 1
    years_sorted = sorted(year_counter.items())
    if years_sorted:
        print(f"\n📅 Rango temporal: {years_sorted[0][0]} — {years_sorted[-1][0]}")

    print("\n" + "=" * 60)
    print("  PIPELINE OK ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
