"""
routes.py -- Rutas FastAPI para el Academic Explorer.

Proporciona tanto paginas HTML (Jinja2) como endpoints JSON.
Carga datos reales OpenAlex + datos persistidos del pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import OUTPUT_DIR, BASE_DIR

logger = logging.getLogger(__name__)

# =====================================================================
# CAPA DE DATOS (pipeline outputs)
# =====================================================================

_profiles_dir = OUTPUT_DIR / "profiles"
_graph_path = OUTPUT_DIR / "knowledge_graph.json"
_refinement_path = OUTPUT_DIR / "refinement_results.json"

_profiles_cache: dict[str, dict] = {}
_graph_cache: Optional[dict] = None
_refinement_cache: list[dict] = []
_retriever = None


def _load_profiles() -> dict[str, dict]:
    global _profiles_cache
    if _profiles_cache:
        return _profiles_cache
    _profiles_cache = {}
    if _profiles_dir.exists():
        for f in _profiles_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    pid = data.get("id", f.stem)
                    _profiles_cache[pid] = data
            except Exception as e:
                logger.warning(f"Error loading profile {f}: {e}")
    return _profiles_cache


def _load_graph() -> dict:
    global _graph_cache
    if _graph_cache:
        return _graph_cache
    if _graph_path.exists():
        with open(_graph_path, "r", encoding="utf-8") as f:
            _graph_cache = json.load(f)
    else:
        _graph_cache = {"nodes": [], "edges": [], "stats": {}}
    return _graph_cache


def _load_refinement() -> list[dict]:
    global _refinement_cache
    if _refinement_cache:
        return _refinement_cache
    if _refinement_path.exists():
        with open(_refinement_path, "r", encoding="utf-8") as f:
            _refinement_cache = json.load(f)
    return _refinement_cache


def _load_retriever():
    global _retriever
    try:
        from src.rag.hybrid_retriever import HybridRetriever
        from src.graph.graph_enrichment import AcademicGraphBuilder
        from src.rag.graph_aware_retriever import GraphAwareRetriever

        hybrid = HybridRetriever()
        graph_builder = AcademicGraphBuilder.load()

        if graph_builder.graph.G.number_of_nodes() == 0 and hybrid.profiles:
            logger.info("Construyendo Knowledge Graph en memoria...")
            graph_builder.build_from_profiles(hybrid.profiles)

        _retriever = GraphAwareRetriever(hybrid, graph_builder)
        if _retriever.load():
            logger.info(f"Retriever cargado: {_retriever.stats()}")
        else:
            logger.info("Retriever en modo fallback (sin índices/grafo)")
    except Exception as e:
        logger.warning(f"No se pudo cargar retriever: {e}")
        _retriever = None


def reload_data():
    global _profiles_cache, _graph_cache, _refinement_cache
    _profiles_cache = {}
    _graph_cache = None
    _refinement_cache = []
    _load_profiles()
    _load_graph()
    _load_refinement()
    _load_retriever()


def _render(request: Request, template: str, ctx: dict, status_code: int = 200):
    ctx["request"] = request
    return templates.TemplateResponse(
        request=request,
        name=template,
        context=ctx,
        status_code=status_code,
    )


# =====================================================================
# FASTAPI APP
# =====================================================================

app = FastAPI(
    title="Modelo-ORCID Academic Explorer",
    description="Explorador de identidades academicas con datos reales OpenAlex",
    version="1.0.0",
)

_api_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_api_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(_api_dir / "templates"))


@app.on_event("startup")
async def startup():
    reload_data()
    # Load OpenAlex data
    from api.openalex_data import get_authors, get_all_works
    authors = get_authors()
    works = get_all_works()
    logger.info(
        f"Startup: {len(_profiles_cache)} pipeline profiles, "
        f"{len(authors)} OpenAlex authors, {len(works)} works"
    )


# =====================================================================
# HTML PAGES
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Pagina principal con buscador y estadisticas reales."""
    from api.openalex_data import get_real_stats, get_authors

    stats = get_real_stats()
    authors = get_authors()

    # Build researcher cards from OpenAlex data
    author_cards = []
    for a in authors:
        last_inst = ""
        lki = a.get("last_known_institutions", [])
        if lki and isinstance(lki, list) and lki:
            last_inst = lki[0].get("display_name", "")

        author_cards.append({
            "slug": a.get("_slug", ""),
            "display_name": a.get("display_name", ""),
            "institution": last_inst,
            "works_count": a.get("works_count", 0),
            "cited_by_count": a.get("cited_by_count", 0),
            "h_index": a.get("summary_stats", {}).get("h_index", 0),
            "topics": [t.get("display_name", t.get("name", "")) for t in a.get("topics", [])[:3]],
            "orcid": a.get("orcid", ""),
        })

    return _render(request, "index.html", {
        "stats": stats,
        "authors": author_cards,
    })


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = Query("", min_length=0)):
    """Pagina de busqueda con datos reales OpenAlex."""
    from api.openalex_data import search_openalex

    results = {"authors": [], "papers": [], "topics": [], "institutions": []}
    if q:
        results = search_openalex(q)

    return _render(request, "search.html", {
        "query": q,
        "results": results,
        "total": (
            len(results["authors"]) + len(results["papers"]) +
            len(results["topics"]) + len(results["institutions"])
        ),
    })


@app.get("/researcher/{researcher_slug}", response_class=HTMLResponse)
async def researcher_profile(request: Request, researcher_slug: str):
    """Perfil de un investigador con datos reales OpenAlex."""
    from api.openalex_data import get_author_by_slug, get_works_for_author

    author = get_author_by_slug(researcher_slug)

    if not author:
        return _render(request, "404.html", {
            "message": f"Investigador '{researcher_slug}' no encontrado",
        }, status_code=404)

    works = get_works_for_author(researcher_slug)

    # Sort works by citations
    works_sorted = sorted(works, key=lambda w: w.get("cited_by_count", 0), reverse=True)

    # Extract all unique topics from works
    work_topics: dict[str, int] = {}
    for w in works:
        for t in w.get("topics", []):
            tname = t.get("name", "")
            if tname:
                work_topics[tname] = work_topics.get(tname, 0) + 1

    # Extract all unique concepts from works
    work_concepts: dict[str, int] = {}
    for w in works:
        for c in w.get("concepts", []):
            cname = c.get("name", "")
            if cname:
                work_concepts[cname] = work_concepts.get(cname, 0) + 1

    # Top concepts sorted by frequency
    top_concepts = sorted(work_concepts.items(), key=lambda x: -x[1])[:20]

    # All affiliations with years
    affiliations = []
    for aff in author.get("affiliations", []):
        inst = aff.get("institution", {})
        affiliations.append({
            "name": inst.get("display_name", ""),
            "country": inst.get("country_code", ""),
            "type": inst.get("type", ""),
            "ror": inst.get("ror", ""),
            "years": sorted(aff.get("years", []), reverse=True),
        })

    # Counts by year for chart
    counts_by_year = author.get("counts_by_year", [])

    # Last known institution
    last_inst = ""
    lki = author.get("last_known_institutions", [])
    if lki and isinstance(lki, list) and lki:
        last_inst = lki[0].get("display_name", "")

    return _render(request, "profile.html", {
        "author": author,
        "works": works_sorted[:50],
        "total_works_loaded": len(works),
        "work_topics": sorted(work_topics.items(), key=lambda x: -x[1])[:15],
        "top_concepts": top_concepts,
        "affiliations": affiliations,
        "counts_by_year": json.dumps(counts_by_year),
        "last_institution": last_inst,
    })


@app.get("/graph", response_class=HTMLResponse)
async def graph_explorer(request: Request):
    """Explorador interactivo del grafo real."""
    from api.openalex_data import build_real_graph
    graph = build_real_graph()
    return _render(request, "graph.html", {
        "graph_data": json.dumps(graph),
        "stats": graph.get("stats", {}),
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard con metricas REALES de OpenAlex."""
    from api.openalex_data import get_real_stats, get_authors

    stats = get_real_stats()
    authors = get_authors()

    # Author overview for table
    author_rows = []
    for a in authors:
        last_inst = ""
        lki = a.get("last_known_institutions", [])
        if lki and isinstance(lki, list) and lki:
            last_inst = lki[0].get("display_name", "")

        author_rows.append({
            "slug": a.get("_slug", ""),
            "name": a.get("display_name", ""),
            "works": a.get("works_count", 0),
            "cited": a.get("cited_by_count", 0),
            "h_index": a.get("summary_stats", {}).get("h_index", 0),
            "institution": last_inst,
        })

    return _render(request, "dashboard.html", {
        "stats": stats,
        "authors": author_rows,
        "papers_by_year_json": json.dumps(stats.get("papers_by_year", {})),
        "top_institutions_json": json.dumps(stats.get("top_institutions", [])),
        "top_topics_json": json.dumps(stats.get("top_topics", [])),
    })


# =====================================================================
# API JSON ENDPOINTS
# =====================================================================

@app.get("/api/stats")
async def api_stats():
    from api.openalex_data import get_real_stats
    return get_real_stats()


@app.get("/api/search")
async def api_search(q: str = Query("", min_length=1), top_k: int = Query(10, ge=1, le=50)):
    from api.openalex_data import search_openalex
    return search_openalex(q)


@app.get("/api/researcher/{slug}")
async def api_researcher(slug: str):
    from api.openalex_data import get_author_by_slug, get_works_for_author
    author = get_author_by_slug(slug)
    if not author:
        return JSONResponse({"error": "Not found"}, status_code=404)
    works = get_works_for_author(slug)
    return {"author": author, "works": works[:20]}


@app.get("/api/graph")
async def api_graph():
    from api.openalex_data import build_real_graph
    return build_real_graph()


@app.post("/api/reload")
async def api_reload():
    reload_data()
    return {"status": "ok", "profiles": len(_profiles_cache)}


# =====================================================================
# INTERPRETABILITY ENDPOINTS (preserved from original)
# =====================================================================

_interp_cache: Optional[dict] = None
_traces_cache: dict[str, dict] = {}


def _load_interpretability() -> dict:
    global _interp_cache
    if _interp_cache:
        return _interp_cache
    interp_path = OUTPUT_DIR / "interpretability_analysis.json"
    if interp_path.exists():
        with open(interp_path, "r", encoding="utf-8") as f:
            _interp_cache = json.load(f)
    else:
        _interp_cache = {}
    return _interp_cache


@app.get("/api/interpretability")
async def api_interpretability():
    return _load_interpretability()
