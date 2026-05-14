"""
routes.py -- Rutas FastAPI para el Academic Explorer.

Proporciona tanto paginas HTML (Jinja2) como endpoints JSON.
Carga datos persistidos del pipeline (profiles, graph, results).
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
from pydantic import BaseModel

from config import OUTPUT_DIR, BASE_DIR

# Import Chat & RAG modules
from src.chat.chat_engine import AcademicChatEngine
from src.rag.chunker import AcademicChunker
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# =====================================================================
# CAPA DE DATOS (carga desde JSON persistidos)
# =====================================================================

_profiles_dir = OUTPUT_DIR / "profiles"
_graph_path = OUTPUT_DIR / "knowledge_graph.json"
_refinement_path = OUTPUT_DIR / "refinement_results.json"

# Cache en memoria
_profiles_cache: dict[str, dict] = {}
_graph_cache: Optional[dict] = None
_refinement_cache: list[dict] = []


def _load_profiles() -> dict[str, dict]:
    """Carga todos los perfiles JSON a memoria."""
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
    """Carga el grafo exportado."""
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
    """Carga resultados de refinamiento."""
    global _refinement_cache
    if _refinement_cache:
        return _refinement_cache

    if _refinement_path.exists():
        with open(_refinement_path, "r", encoding="utf-8") as f:
            _refinement_cache = json.load(f)

    return _refinement_cache


def reload_data():
    """Fuerza recarga de todos los datos."""
    global _profiles_cache, _graph_cache, _refinement_cache
    _profiles_cache = {}
    _graph_cache = None
    _refinement_cache = []
    _load_profiles()
    _load_graph()
    _load_refinement()


def _render(request: Request, template: str, ctx: dict, status_code: int = 200):
    """Helper para renderizar templates compatible con Starlette 1.0+."""
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
    description="Explorador de identidades academicas de investigadores SNII",
    version="0.5.0",
)

# Static files & templates
_api_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_api_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(_api_dir / "templates"))


@app.on_event("startup")
async def startup():
    """Carga datos al iniciar."""
    reload_data()
    logger.info(f"Loaded {len(_profiles_cache)} profiles")
    
    # Init Chat Engine
    global chat_engine
    chat_engine = AcademicChatEngine()


# =====================================================================
# PAGINAS HTML
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Pagina principal con buscador y estadisticas."""
    profiles = _load_profiles()
    graph = _load_graph()

    sorted_profiles = sorted(
        profiles.values(),
        key=lambda p: p.get("confidence", 0),
        reverse=True,
    )

    return _render(request, "index.html", {
        "stats": {
            "researchers": len(profiles),
            "with_orcid": sum(1 for p in profiles.values() if p.get("orcid_id")),
            "institutions": len(set(p.get("institution", "") for p in profiles.values())),
            "graph_nodes": graph.get("stats", {}).get("total_nodes", 0),
            "graph_edges": graph.get("stats", {}).get("total_edges", 0),
        },
        "profiles": sorted_profiles[:12],
    })


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = Query("", min_length=0)):
    """Pagina de busqueda."""
    profiles = _load_profiles()
    results = []

    if q:
        q_lower = q.lower()
        for p in profiles.values():
            name = p.get("full_name", "").lower()
            inst = p.get("institution", "").lower()
            disc = p.get("discipline", "").lower()
            orcid = (p.get("orcid_id") or "").lower()

            if (q_lower in name or q_lower in inst
                    or q_lower in disc or q_lower in orcid):
                results.append(p)

        results.sort(key=lambda p: p.get("confidence", 0), reverse=True)

    return _render(request, "search.html", {
        "query": q,
        "results": results,
        "total": len(results),
    })


@app.get("/researcher/{researcher_id}", response_class=HTMLResponse)
async def researcher_profile(request: Request, researcher_id: str):
    """Pagina de perfil de un investigador."""
    profiles = _load_profiles()
    profile = profiles.get(researcher_id)

    if not profile:
        for pid, p in profiles.items():
            slug = p.get("full_name", "").lower().replace(" ", "_")[:40]
            if slug == researcher_id or pid == researcher_id:
                profile = p
                break

    if not profile:
        return _render(request, "404.html", {
            "message": f"Investigador '{researcher_id}' no encontrado",
        }, status_code=404)

    # Get subgraph for this researcher
    graph = _load_graph()
    snii_node_id = f"snii:{profile.get('id', '')}"

    related_ids = {snii_node_id}
    for edge in graph.get("edges", []):
        if edge["source"] == snii_node_id or edge["target"] == snii_node_id:
            related_ids.add(edge["source"])
            related_ids.add(edge["target"])
    for edge in graph.get("edges", []):
        if edge["source"] in related_ids or edge["target"] in related_ids:
            related_ids.add(edge["source"])
            related_ids.add(edge["target"])

    subgraph_nodes = [n for n in graph.get("nodes", []) if n["id"] in related_ids]
    subgraph_edges = [e for e in graph.get("edges", [])
                      if e["source"] in related_ids and e["target"] in related_ids]

    # Refinement history
    refinement = _load_refinement()
    history = None
    for r in refinement:
        if r.get("researcher_id") == snii_node_id:
            history = r
            break

    return _render(request, "profile.html", {
        "profile": profile,
        "subgraph": json.dumps({"nodes": subgraph_nodes, "edges": subgraph_edges}),
        "history": history,
    })


@app.get("/graph", response_class=HTMLResponse)
async def graph_explorer(request: Request):
    """Explorador interactivo del grafo completo."""
    graph = _load_graph()
    return _render(request, "graph.html", {
        "graph_data": json.dumps(graph),
        "stats": graph.get("stats", {}),
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard con metricas del pipeline."""
    profiles = _load_profiles()
    graph = _load_graph()
    refinement = _load_refinement()

    confidences = [p.get("confidence", 0) for p in profiles.values()]
    conf_buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-1.0": 0}
    for c in confidences:
        if c < 0.3: conf_buckets["0.0-0.3"] += 1
        elif c < 0.5: conf_buckets["0.3-0.5"] += 1
        elif c < 0.7: conf_buckets["0.5-0.7"] += 1
        else: conf_buckets["0.7-1.0"] += 1

    inst_counts: dict[str, int] = {}
    for p in profiles.values():
        inst = p.get("institution", "Unknown")
        inst_counts[inst] = inst_counts.get(inst, 0) + 1

    area_counts: dict[str, int] = {}
    for p in profiles.values():
        area = p.get("area", "").split(" - ")[0].strip() or "Unknown"
        area_counts[area] = area_counts.get(area, 0) + 1

    avg_iter = 0
    if refinement:
        avg_iter = sum(r.get("iterations", 0) for r in refinement) / len(refinement)

    return _render(request, "dashboard.html", {
        "total_profiles": len(profiles),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "with_orcid": sum(1 for p in profiles.values() if p.get("orcid_id")),
        "avg_iterations": avg_iter,
        "conf_buckets": json.dumps(conf_buckets),
        "inst_counts": json.dumps(dict(sorted(inst_counts.items(), key=lambda x: -x[1])[:10])),
        "area_counts": json.dumps(area_counts),
        "graph_stats": graph.get("stats", {}),
    })


# =====================================================================
# API JSON ENDPOINTS
# =====================================================================

@app.get("/api/stats")
async def api_stats():
    """Estadisticas generales."""
    profiles = _load_profiles()
    graph = _load_graph()
    return {
        "total_researchers": len(profiles),
        "with_orcid": sum(1 for p in profiles.values() if p.get("orcid_id")),
        "graph": graph.get("stats", {}),
    }


@app.get("/api/researcher/{researcher_id}")
async def api_researcher(researcher_id: str):
    """Perfil JSON de un investigador."""
    profiles = _load_profiles()
    profile = profiles.get(researcher_id)
    if not profile:
        for pid, p in profiles.items():
            if pid == researcher_id:
                profile = p
                break
    if not profile:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return profile


@app.get("/api/search")
async def api_search(q: str = Query("", min_length=1)):
    """Busqueda JSON."""
    profiles = _load_profiles()
    results = []
    q_lower = q.lower()
    for p in profiles.values():
        name = p.get("full_name", "").lower()
        inst = p.get("institution", "").lower()
        if q_lower in name or q_lower in inst:
            results.append(p)
    return {"query": q, "results": results, "total": len(results)}


@app.get("/api/graph/{researcher_id}")
async def api_graph(researcher_id: str):
    """Subgrafo JSON de un investigador."""
    graph = _load_graph()
    snii_id = f"snii:{researcher_id}"
    related = {snii_id}

    for edge in graph.get("edges", []):
        if edge["source"] == snii_id or edge["target"] == snii_id:
            related.add(edge["source"])
            related.add(edge["target"])

    nodes = [n for n in graph["nodes"] if n["id"] in related]
    edges = [e for e in graph["edges"]
             if e["source"] in related and e["target"] in related]
    return {"nodes": nodes, "edges": edges}


@app.post("/api/reload")
async def api_reload():
    """Recarga datos del disco."""
    reload_data()
    return {"status": "ok", "profiles": len(_profiles_cache)}


# =====================================================================
# FASE 7: INTERPRETABILITY API ENDPOINTS
# =====================================================================

_interp_cache: Optional[dict] = None
_traces_cache: dict[str, dict] = {}


def _load_interpretability() -> dict:
    """Carga datos de analisis de interpretabilidad."""
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


def _load_trace(researcher_id: str) -> Optional[dict]:
    """Carga traza de evidencia de un investigador."""
    if researcher_id in _traces_cache:
        return _traces_cache[researcher_id]

    trace_id = f"snii_{researcher_id}"
    traces_dir = OUTPUT_DIR / "traces"
    trace_path = traces_dir / f"{trace_id}_trace.json"
    if trace_path.exists():
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _traces_cache[researcher_id] = data
            return data
    return None


def _load_viz_data(researcher_id: str) -> Optional[dict]:
    """Carga datos de visualizacion de un investigador."""
    figures_dir = Path("data/reports/figures")
    viz_path = figures_dir / f"snii_{researcher_id}_viz.json"
    if viz_path.exists():
        with open(viz_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@app.get("/api/interpretability")
async def api_interpretability():
    """Analisis completo de interpretabilidad."""
    return _load_interpretability()


@app.get("/api/trace/{researcher_id}")
async def api_trace(researcher_id: str):
    """Traza de evidencia de un investigador."""
    trace = _load_trace(researcher_id)
    if not trace:
        return JSONResponse({"error": "Trace not found"}, status_code=404)
    return trace


@app.get("/api/explanation/{researcher_id}")
async def api_explanation(researcher_id: str):
    """Explicacion de un investigador."""
    interp = _load_interpretability()
    explanations = interp.get("explanations", [])
    snii_id = f"snii:{researcher_id}"
    for e in explanations:
        if e.get("researcher_name", ""):
            # Match by profile id through profiles
            profiles = _load_profiles()
            profile = profiles.get(researcher_id)
            if profile and e["researcher_name"] == profile.get("full_name", ""):
                return e
    return JSONResponse({"error": "Explanation not found"}, status_code=404)


@app.get("/api/viz/{researcher_id}")
async def api_viz(researcher_id: str):
    """Datos de visualizacion de un investigador."""
    data = _load_viz_data(researcher_id)
    if not data:
        return JSONResponse({"error": "Visualization data not found"}, status_code=404)
    return data


@app.get("/api/math-mapping")
async def api_math_mapping():
    """Tabla de mapeo matematico."""
    try:
        from src.math import MathematicalMapping
        mapping = MathematicalMapping()
        return {
            "mappings": mapping.get_mapping_table(),
            "latex": mapping.export_latex_table(),
        }
    except Exception as e:
        return {"error": str(e)}


# =====================================================================
# INTERPRETABILITY HTML PAGE
# =====================================================================

@app.get("/interpretability/{researcher_id}", response_class=HTMLResponse)
async def interpretability_page(request: Request, researcher_id: str):
    """Pagina de interpretabilidad para un investigador."""
    profiles = _load_profiles()
    profile = profiles.get(researcher_id)

    if not profile:
        return _render(request, "404.html", {
            "message": f"Investigador '{researcher_id}' no encontrado",
        }, status_code=404)

    # Load trace
    trace = _load_trace(researcher_id)

    # Load visualization data
    viz_data = _load_viz_data(researcher_id)

    # Load explanation
    interp = _load_interpretability()
    explanation = None
    for e in interp.get("explanations", []):
        if e.get("researcher_name", "") == profile.get("full_name", ""):
            explanation = e
            break

    # Load ambiguity report
    ambiguity = None
    for r in interp.get("ambiguity", {}).get("reports", []):
        if r.get("researcher_name", "") == profile.get("full_name", ""):
            ambiguity = r
            break

    # Load dynamics report
    dynamics = None
    for d in interp.get("dynamics", []):
        if d.get("researcher_name", "") == profile.get("full_name", ""):
            dynamics = d
            break

    return _render(request, "interpretability.html", {
        "profile": profile,
        "trace": json.dumps(trace) if trace else "null",
        "viz_data": json.dumps(viz_data) if viz_data else "null",
        "explanation": explanation,
        "ambiguity": ambiguity,
        "dynamics": dynamics,
        "math_mapping": interp.get("verifications", []),
    })

# =====================================================================
# FASE 8: CONVERSATIONAL CHAT & RAG
# =====================================================================

class ChatMessage(BaseModel):
    message: str

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Interfaz del Asistente Conversacional."""
    return _render(request, "chat.html", {})

@app.post("/api/chat")
async def api_chat(data: ChatMessage):
    """Procesa un mensaje de chat usando el AcademicChatEngine (Ollama)."""
    global chat_engine
    if not chat_engine:
        chat_engine = AcademicChatEngine()
        
    response, sources = chat_engine.process_message(data.message)
    return {"response": response, "sources": sources}

@app.post("/api/rag/index")
async def api_rag_index():
    """Re-indexa los perfiles actuales en el Vector Store (FAISS)."""
    profiles = _load_profiles()
    chunker = AcademicChunker()
    vector_store = VectorStore()
    
    all_chunks = []
    for pid, p in profiles.items():
        all_chunks.extend(chunker.chunk_profile(p))
        
        # Opcional: chunk de explainability si existe
        trace = _load_trace(pid)
        if trace:
            # Simplificamos para no complicar el payload
            exp_dict = {"confidence_score": p.get("confidence", 0)}
            all_chunks.extend(chunker.chunk_explanation(pid, p.get("full_name", ""), exp_dict))
            
    vector_store.add_chunks(all_chunks)
    vector_store.save()
    
    return {"status": "ok", "chunks_indexed": len(all_chunks)}

