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
_unam_directory_cache: dict[str, dict] = {}
_retriever = None


def _load_profiles() -> dict[str, dict]:
    global _profiles_cache
    if _profiles_cache:
        return _profiles_cache
    
    _profiles_cache = {}
    from config import PROCESSED_DATA_DIR
    unam_path = PROCESSED_DATA_DIR / "unam_authors.json"
    
    if unam_path.exists():
        try:
            with open(unam_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for profile in data:
                    pid = profile.get("id")
                    if pid:
                        _profiles_cache[pid] = profile
        except Exception as e:
            logger.warning(f"Error loading UNAM profiles for cache: {e}")
            
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

def _load_unam_directory() -> dict[str, dict]:
    global _unam_directory_cache
    if _unam_directory_cache:
        return _unam_directory_cache
    
    from config import PROCESSED_DATA_DIR
    unam_path = PROCESSED_DATA_DIR / "unam_directory.json"
    if unam_path.exists():
        try:
            with open(unam_path, "r", encoding="utf-8") as f:
                _unam_directory_cache = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading UNAM directory: {e}")
    return _unam_directory_cache


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
    global _profiles_cache, _graph_cache, _refinement_cache, _unam_directory_cache
    _profiles_cache = {}
    _graph_cache = None
    _refinement_cache = []
    _unam_directory_cache = {}
    _load_profiles()
    _load_graph()
    _load_refinement()
    _load_unam_directory()
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
            "topics": [t if isinstance(t, str) else t.get("display_name", t.get("name", "")) for t in a.get("topics", [])[:3]],
            "orcid": a.get("orcid", ""),
        })

    return _render(request, "index.html", {
        "stats": stats,
        "authors": author_cards,
    })


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = Query("", min_length=0)):
    """Pagina de busqueda con motor de retrieval."""
    from api.openalex_data import search_openalex

    results = {"authors": [], "papers": [], "topics": [], "institutions": []}
    if q:
        if _retriever:
            # Flujo REAL: request -> routes.py -> retriever -> query interpreter -> FAISS/BM25
            raw_results = _retriever.search(q, top_k=20)
            print(f"\n[ROUTES] _retriever.search returned: type={type(raw_results)}, len={len(raw_results)}")
            if raw_results:
                print(f"[ROUTES] raw_results[0].keys() = {raw_results[0].keys()}")
                print(f"[ROUTES] raw_results[0] = {raw_results[0]}")
            for r in raw_results:
                node_type = r.get("node_type")
                is_paper = r.get("id", "").startswith("work_")
                
                if node_type == "paper" or is_paper:
                    results["papers"].append({
                        "title": r.get("nombre_completo", r.get("display_name", "")),
                        "year": r.get("year", ""),
                        "citations": r.get("citations", 0),
                        "doi": r.get("doi", ""),
                        "venue": r.get("venue", ""),
                        "author": r.get("author", ""),
                        "institutions": r.get("institutions", []),
                        "topics": r.get("topics", []),
                        "concepts": r.get("concepts", []),
                        "score": r.get("score"),
                        "explanation": r.get("explanation", ""),
                    })
                elif node_type == "researcher" or not is_paper:
                    author_dict = {
                        "slug": r.get("id"),
                        "display_name": r.get("nombre_completo", r.get("display_name", "")),
                        "works_count": r.get("works_count", 0),
                        "cited_by_count": r.get("cited_by_count", 0),
                        "h_index": r.get("h_index", 0),
                        "institution": r.get("institucion", r.get("institution", "")),
                        "topics": [r.get("disciplina", "")] if r.get("disciplina") else r.get("topics", []),
                        "orcid": r.get("orcid", ""),
                        "score": r.get("score"),
                        "explanation": r.get("explanation", ""),
                    }
                    
                    import unicodedata
                    n_raw = author_dict["display_name"].strip().lower()
                    nfkd = unicodedata.normalize("NFKD", n_raw)
                    norm_name = "".join(c for c in nfkd if not unicodedata.combining(c))
                    
                    _load_unam_directory()
                    if norm_name in _unam_directory_cache:
                        author_dict["unam_source"] = _unam_directory_cache[norm_name]
                        
                    results["authors"].append(author_dict)
                elif node_type == "topic":
                    results["topics"].append(r)
                elif node_type == "institution":
                    results["institutions"].append(r)
            print(f"[ROUTES] mapped: authors={len(results['authors'])}, papers={len(results['papers'])}, topics={len(results['topics'])}, institutions={len(results['institutions'])}")
        else:
            # Fallback a OpenAlex in-memory si el retriever falló al cargar
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

    # 🚀 ENRIQUECIMIENTO BAJO DEMANDA (OpenAlex)
    works = get_works_for_author(researcher_slug)
    
    if author.get("openalex_id"):
        try:
            from src.enrichment.openalex_enricher import OpenAlexEnricher
            enricher = OpenAlexEnricher()
            # Esto enriquecerá author con metrics, topics y works si están en cache o via API
            author = enricher.enrich_profile(author)
            
            # Si el enriquecedor trajo nuevos trabajos, usarlos
            if "works" in author and author["works"]:
                works = author["works"]
                logger.info(f"✨ Perfil enriquecido para {researcher_slug} ({len(works)} trabajos)")
        except Exception as e:
            logger.warning(f"⚠️ Error enriqueciendo perfil {researcher_slug}: {e}")

    # Sort works by citations
    works_sorted = sorted(works, key=lambda w: w.get("cited_by_count", 0), reverse=True)

    # Extract all unique topics from works
    work_topics: dict[str, int] = {}
    for w in works:
        for t in w.get("topics", []):
            tname = t if isinstance(t, str) else t.get("name", "")
            if tname:
                work_topics[tname] = work_topics.get(tname, 0) + 1

    # Extract all unique concepts from works
    work_concepts: dict[str, int] = {}
    for w in works:
        for c in w.get("concepts", []):
            cname = c if isinstance(c, str) else c.get("name", "")
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

    # Check for extra sources (UNAM, etc)
    unam_dir = _load_unam_directory()
    # Normalize name to match
    import unicodedata
    def _norm(text):
        if not text: return ""
        nfkd = unicodedata.normalize("NFKD", text.lower().strip())
        return "".join(c for c in nfkd if not unicodedata.combining(c))
    
    author_name_norm = _norm(author.get("display_name", ""))
    extra_sources = []
    if author_name_norm in unam_dir:
        extra_sources.append({
            "name": "Directorio UNAM",
            "url": unam_dir[author_name_norm].get("url_fuente", "#"),
            "info": unam_dir[author_name_norm].get("departamento", "UNAM")
        })
    
    # Check for ORCID as an extra source too if present
    if author.get("orcid"):
        extra_sources.append({
            "name": "ORCID Public Profile",
            "url": author["orcid"],
            "info": author["orcid"].split('/')[-1]
        })

    return _render(request, "profile.html", {
        "author": author,
        "works": works_sorted[:50],
        "total_works_loaded": len(works),
        "work_topics": sorted(work_topics.items(), key=lambda x: -x[1])[:15],
        "top_concepts": top_concepts,
        "affiliations": affiliations,
        "counts_by_year": json.dumps(counts_by_year),
        "last_institution": last_inst,
        "extra_sources": extra_sources,
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
    if _retriever:
        return _retriever.search(q, top_k=top_k)
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
