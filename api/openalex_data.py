"""
openalex_data.py -- Carga y sirve datos REALES de OpenAlex para el portal.

Lee directamente los archivos JSON en data/openalex/ y construye
índices en memoria para búsqueda, dashboard y grafo.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from config import BASE_DIR

logger = logging.getLogger(__name__)

OPENALEX_DIR = BASE_DIR / "data" / "openalex"

# ─── In-memory caches ──────────────────────────────────────────────────
_authors: list[dict] = []
_works: list[dict] = []          # flat list of all works
_works_by_author: dict[str, list[dict]] = {}   # author_name -> [works]
_all_topics: list[dict] = []      # unique topics across everything
_all_institutions: list[dict] = []  # unique institutions
_stats_cache: Optional[dict] = None
_graph_cache: Optional[dict] = None


def _load_all():
    """Carga todos los datos OpenAlex a memoria."""
    global _authors, _works, _works_by_author, _all_topics, _all_institutions
    global _stats_cache, _graph_cache

    if _authors:
        return  # already loaded

    authors_dir = OPENALEX_DIR / "authors"
    works_dir = OPENALEX_DIR / "works"

    # ── Load authors ─────────────────────────────────────────────────
    if authors_dir.exists():
        for fp in sorted(authors_dir.glob("*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    author = json.load(f)
                    # Create a stable slug-id from the filename
                    author["_slug"] = fp.stem  # e.g. "Miguel_Alcubierre"
                    _authors.append(author)
            except Exception as e:
                logger.warning(f"Error loading author {fp}: {e}")

    # ── Load works ───────────────────────────────────────────────────
    if works_dir.exists():
        for fp in sorted(works_dir.glob("*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    works_list = json.load(f)
                    # Author name from filename: "Miguel_Alcubierre_works.json" -> "Miguel_Alcubierre"
                    author_key = fp.stem.replace("_works", "")
                    _works_by_author[author_key] = works_list
                    for w in works_list:
                        w["_author_key"] = author_key
                    _works.extend(works_list)
            except Exception as e:
                logger.warning(f"Error loading works {fp}: {e}")

    # ── Build topics index ───────────────────────────────────────────
    seen_topic_ids = set()
    for a in _authors:
        for t in a.get("topics", []):
            tid = t.get("id", "")
            if tid and tid not in seen_topic_ids:
                seen_topic_ids.add(tid)
                _all_topics.append({
                    "id": tid,
                    "name": t.get("display_name", t.get("name", "")),
                    "count": t.get("count", 0),
                    "subfield": t.get("subfield", {}).get("display_name", ""),
                    "field": t.get("field", {}).get("display_name", ""),
                    "domain": t.get("domain", {}).get("display_name", ""),
                })

    # Also collect topics from works
    for w in _works:
        for t in w.get("topics", []):
            tid = t.get("id", "")
            if tid and tid not in seen_topic_ids:
                seen_topic_ids.add(tid)
                _all_topics.append({
                    "id": tid,
                    "name": t.get("name", ""),
                    "count": 1,
                    "subfield": "",
                    "field": "",
                    "domain": "",
                })

    # ── Build institutions index ─────────────────────────────────────
    seen_inst = set()
    for a in _authors:
        for aff in a.get("affiliations", []):
            inst = aff.get("institution", {})
            iname = inst.get("display_name", "")
            if iname and iname not in seen_inst:
                seen_inst.add(iname)
                _all_institutions.append({
                    "name": iname,
                    "ror": inst.get("ror", ""),
                    "country": inst.get("country_code", ""),
                    "type": inst.get("type", ""),
                })
    # Also from works
    for w in _works:
        for iname in w.get("institutions", []):
            if iname and iname not in seen_inst:
                seen_inst.add(iname)
                _all_institutions.append({
                    "name": iname,
                    "ror": "",
                    "country": "",
                    "type": "",
                })

    logger.info(
        f"OpenAlex loaded: {len(_authors)} authors, {len(_works)} works, "
        f"{len(_all_topics)} topics, {len(_all_institutions)} institutions"
    )


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def get_authors() -> list[dict]:
    _load_all()
    return _authors


def get_author_by_slug(slug: str) -> Optional[dict]:
    """Busca un autor por su slug (filename stem)."""
    _load_all()
    for a in _authors:
        if a.get("_slug") == slug:
            return a
    # Try partial match
    slug_lower = slug.lower().replace("-", "_").replace(" ", "_")
    for a in _authors:
        if a.get("_slug", "").lower() == slug_lower:
            return a
    return None

def get_author(author_id_or_slug: str) -> Optional[dict]:
    """Alias para buscar autor por ID o slug."""
    _load_all()
    # Primero intentar por slug
    a = get_author_by_slug(author_id_or_slug)
    if a: return a
    # Buscar por id
    for a in _authors:
        if a.get("id") == author_id_or_slug or a.get("id", "").split("/")[-1] == author_id_or_slug:
            return a
    return None


def get_works_for_author(slug: str) -> list[dict]:
    _load_all()
    return _works_by_author.get(slug, [])


def get_all_works() -> list[dict]:
    _load_all()
    return _works


def get_work(work_id_or_doi: str) -> Optional[dict]:
    """Obtiene un work por su ID o DOI."""
    _load_all()
    for w in _works:
        if w.get("openalex_id") == work_id_or_doi or w.get("openalex_id", "").split("/")[-1] == work_id_or_doi:
            return w
        if w.get("doi") == work_id_or_doi or w.get("doi", "").replace("https://doi.org/", "") == work_id_or_doi:
            return w
    return None


def get_all_topics() -> list[dict]:
    _load_all()
    return _all_topics


def get_all_institutions() -> list[dict]:
    _load_all()
    return _all_institutions


def get_real_stats() -> dict:
    """Computes comprehensive stats from real OpenAlex data."""
    global _stats_cache
    _load_all()

    if _stats_cache:
        return _stats_cache

    total_citations = sum(a.get("cited_by_count", 0) for a in _authors)
    total_works_count = sum(a.get("works_count", 0) for a in _authors)

    # Papers by year
    year_counts: Counter = Counter()
    for w in _works:
        yr = w.get("publication_year")
        if yr:
            year_counts[yr] += 1

    # Top cited papers
    sorted_works = sorted(_works, key=lambda w: w.get("cited_by_count", 0), reverse=True)
    top_cited = []
    for w in sorted_works[:10]:
        top_cited.append({
            "title": w.get("title", "")[:100],
            "year": w.get("publication_year"),
            "citations": w.get("cited_by_count", 0),
            "doi": w.get("doi", ""),
            "author": w.get("_author_key", "").replace("_", " "),
        })

    # Top institutions (by frequency in works)
    inst_counter: Counter = Counter()
    for w in _works:
        for inst in w.get("institutions", []):
            inst_counter[inst] += 1

    top_institutions = [
        {"name": name, "count": count}
        for name, count in inst_counter.most_common(15)
    ]

    # Top topics
    topic_counter: Counter = Counter()
    for w in _works:
        for t in w.get("topics", []):
            topic_counter[t.get("name", "")] += 1

    top_topics = [
        {"name": name, "count": count}
        for name, count in topic_counter.most_common(15)
    ]

    _stats_cache = {
        "total_authors": len(_authors),
        "total_works": len(_works),
        "total_works_count": total_works_count,
        "total_topics": len(_all_topics),
        "total_institutions": len(_all_institutions),
        "total_citations": total_citations,
        "papers_by_year": dict(sorted(year_counts.items())),
        "top_cited_papers": top_cited,
        "top_institutions": top_institutions,
        "top_topics": top_topics,
    }
    return _stats_cache


def get_top_papers(limit: int = 10) -> list[dict]:
    """Obtiene los papers mas citados."""
    stats = get_real_stats()
    return stats.get("top_cited_papers", [])[:limit]


def get_top_institutions(limit: int = 10) -> list[dict]:
    """Obtiene las instituciones mas frecuentes."""
    stats = get_real_stats()
    return stats.get("top_institutions", [])[:limit]


def search_topics(query: str) -> list[dict]:
    """Busca topics especificamente."""
    _load_all()
    q = query.lower().strip()
    if not q:
        return []
    return [t for t in _all_topics if q in t.get("name", "").lower()]


def search_openalex(query: str) -> dict:
    """Busca en los datos de OpenAlex: autores, papers, topics."""
    _load_all()
    q = query.lower().strip()
    if not q:
        return {"authors": [], "papers": [], "topics": [], "institutions": []}

    # ── Search authors ─────────────────────────────────────────────
    matched_authors = []
    for a in _authors:
        name = a.get("display_name", "").lower()
        topics_str = " ".join(
            t.get("display_name", t.get("name", ""))
            for t in a.get("topics", [])
        ).lower()
        concepts_str = " ".join(
            c.get("display_name", "")
            for c in a.get("x_concepts", [])
        ).lower()
        inst_str = " ".join(
            aff.get("institution", {}).get("display_name", "")
            for aff in a.get("affiliations", [])
        ).lower()

        score = 0
        if q in name:
            score += 100
        if q in topics_str:
            score += 40
        if q in concepts_str:
            score += 30
        if q in inst_str:
            score += 20

        if score > 0:
            # Build explanation
            explanations = []
            if q in name:
                explanations.append(f"Nombre coincide con \"{query}\"")
            if q in topics_str:
                # Find matching topic
                for t in a.get("topics", []):
                    tn = t.get("display_name", t.get("name", "")).lower()
                    if q in tn:
                        explanations.append(f"Relacionado mediante topic: {t.get('display_name', t.get('name', ''))}")
                        break
            if q in concepts_str:
                for c in a.get("x_concepts", []):
                    cn = c.get("display_name", "").lower()
                    if q in cn:
                        explanations.append(f"Concepto relacionado: {c.get('display_name', '')}")
                        break
            if q in inst_str:
                for aff in a.get("affiliations", []):
                    iname = aff.get("institution", {}).get("display_name", "").lower()
                    if q in iname:
                        explanations.append(f"Coincidencia por institución: {aff['institution']['display_name']}")
                        break

            last_inst = ""
            lki = a.get("last_known_institutions", [])
            if lki and isinstance(lki, list) and lki:
                last_inst = lki[0].get("display_name", "")

            matched_authors.append({
                "slug": a.get("_slug", ""),
                "display_name": a.get("display_name", ""),
                "works_count": a.get("works_count", 0),
                "cited_by_count": a.get("cited_by_count", 0),
                "h_index": a.get("summary_stats", {}).get("h_index", 0),
                "institution": last_inst,
                "topics": [t.get("display_name", t.get("name", "")) for t in a.get("topics", [])[:5]],
                "orcid": a.get("orcid", ""),
                "score": score,
                "explanation": " · ".join(explanations),
            })

    matched_authors.sort(key=lambda x: x["score"], reverse=True)

    # ── Search papers ──────────────────────────────────────────────
    matched_papers = []
    for w in _works:
        title = (w.get("title") or "").lower()
        abstract = (w.get("abstract") or "").lower()
        concepts_str = " ".join(c.get("name", "") for c in w.get("concepts", [])).lower()
        topics_str = " ".join(t.get("name", "") for t in w.get("topics", [])).lower()

        score = 0
        if q in title:
            score += 80
        if q in abstract:
            score += 50
        if q in topics_str:
            score += 30
        if q in concepts_str:
            score += 20

        if score > 0:
            explanation_parts = []
            if q in title:
                explanation_parts.append("Título coincide")
            if q in abstract:
                explanation_parts.append("Coincidencia en abstract")
            if q in topics_str:
                for t in w.get("topics", []):
                    if q in t.get("name", "").lower():
                        explanation_parts.append(f"Topic: {t['name']}")
                        break
            if q in concepts_str:
                for c in w.get("concepts", []):
                    if q in c.get("name", "").lower():
                        explanation_parts.append(f"Concepto: {c['name']}")
                        break

            matched_papers.append({
                "title": w.get("title", ""),
                "year": w.get("publication_year"),
                "citations": w.get("cited_by_count", 0),
                "doi": w.get("doi", ""),
                "venue": w.get("venue", ""),
                "author": w.get("_author_key", "").replace("_", " "),
                "institutions": w.get("institutions", [])[:5],
                "topics": [t.get("name", "") for t in w.get("topics", [])[:3]],
                "concepts": [c.get("name", "") for c in w.get("concepts", [])[:5]],
                "score": score,
                "explanation": " · ".join(explanation_parts),
            })

    matched_papers.sort(key=lambda x: x["score"], reverse=True)

    # ── Search topics ──────────────────────────────────────────────
    matched_topics = []
    for t in _all_topics:
        tname = t.get("name", "").lower()
        if q in tname:
            matched_topics.append(t)

    # ── Search institutions ────────────────────────────────────────
    matched_institutions = []
    for inst in _all_institutions:
        if q in inst.get("name", "").lower():
            matched_institutions.append(inst)

    return {
        "authors": matched_authors[:20],
        "papers": matched_papers[:20],
        "topics": matched_topics[:20],
        "institutions": matched_institutions[:20],
    }


def build_real_graph() -> dict:
    """Construye un grafo real con nodos y aristas de OpenAlex."""
    global _graph_cache
    _load_all()

    if _graph_cache:
        return _graph_cache

    nodes = []
    edges = []
    node_ids = set()

    # Colors
    COLORS = {
        "author": "#3b82f6",
        "paper": "#10b981",
        "institution": "#8b5cf6",
        "topic": "#f59e0b",
    }

    # ── Author nodes ────────────────────────────────────────────────
    for a in _authors:
        aid = f"author:{a['_slug']}"
        nodes.append({
            "id": aid,
            "label": a.get("display_name", ""),
            "type": "author",
            "color": COLORS["author"],
            "works_count": a.get("works_count", 0),
            "cited_by_count": a.get("cited_by_count", 0),
            "size": 30,
        })
        node_ids.add(aid)

        # ── Institution edges ──────────────────────────────────────
        lki = a.get("last_known_institutions", [])
        if lki and isinstance(lki, list):
            for inst in lki[:2]:
                iname = inst.get("display_name", "")
                iid = f"inst:{iname[:50].lower().replace(' ', '_')}"
                if iid not in node_ids:
                    nodes.append({
                        "id": iid,
                        "label": iname,
                        "type": "institution",
                        "color": COLORS["institution"],
                        "country": inst.get("country_code", ""),
                        "size": 20,
                    })
                    node_ids.add(iid)
                edges.append({
                    "source": aid,
                    "target": iid,
                    "type": "affiliated_with",
                })

        # ── Topic edges ────────────────────────────────────────────
        for t in a.get("topics", [])[:3]:
            tname = t.get("display_name", t.get("name", ""))
            tid = f"topic:{tname[:50].lower().replace(' ', '_')}"
            if tid not in node_ids:
                nodes.append({
                    "id": tid,
                    "label": tname,
                    "type": "topic",
                    "color": COLORS["topic"],
                    "size": 12,
                })
                node_ids.add(tid)
            edges.append({
                "source": aid,
                "target": tid,
                "type": "related_to_topic",
            })

    # ── Paper nodes (top cited per author) ──────────────────────────
    for author_key, works_list in _works_by_author.items():
        aid = f"author:{author_key}"
        top_works = sorted(works_list, key=lambda w: w.get("cited_by_count", 0), reverse=True)[:5]
        for w in top_works:
            oid = w.get("openalex_id", "")
            short = oid.split("/")[-1] if "/" in oid else oid
            pid = f"paper:{short}"
            if pid not in node_ids:
                nodes.append({
                    "id": pid,
                    "label": (w.get("title", "") or "")[:40],
                    "type": "paper",
                    "color": COLORS["paper"],
                    "year": w.get("publication_year"),
                    "citations": w.get("cited_by_count", 0),
                    "size": min(8 + (w.get("cited_by_count", 0) or 0) // 50, 25),
                })
                node_ids.add(pid)
            edges.append({
                "source": aid,
                "target": pid,
                "type": "authored",
            })

            # Citation edges between papers we have
            for ref_id in (w.get("referenced_works") or [])[:5]:
                ref_short = ref_id.split("/")[-1] if "/" in ref_id else ref_id
                ref_pid = f"paper:{ref_short}"
                if ref_pid in node_ids:
                    edges.append({
                        "source": pid,
                        "target": ref_pid,
                        "type": "cites",
                    })

    _graph_cache = {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "authors": sum(1 for n in nodes if n["type"] == "author"),
            "papers": sum(1 for n in nodes if n["type"] == "paper"),
            "institutions": sum(1 for n in nodes if n["type"] == "institution"),
            "topics": sum(1 for n in nodes if n["type"] == "topic"),
        },
    }
    return _graph_cache
