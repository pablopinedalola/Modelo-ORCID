"""
graph_aware_retriever.py — Motor de búsqueda basado en Knowledge Graphs.

Extiende HybridRetriever añadiendo:
1. Retrieval Inicial Híbrido (FAISS + BM25).
2. Mapeo a nodos de grafo.
3. Evidence Propagation a vecinos relevantes.
4. Reranking considerando la proximidad de red.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.rag.hybrid_retriever import HybridRetriever
from src.graph.graph_enrichment import AcademicGraphBuilder
from src.graph.neighbor_retriever import NeighborRetriever
from src.refinement.evidence_propagation import EvidencePropagation
from src.models.schemas import NodeType

logger = logging.getLogger(__name__)

class GraphAwareRetriever:
    """Motor de búsqueda híbrido + propagación de grafos.
    
    Combina: Semántica + Léxica + Propagación de Grafo.
    """

    def __init__(self, hybrid_retriever: HybridRetriever, graph_builder: AcademicGraphBuilder):
        self.hybrid_retriever = hybrid_retriever
        self.graph_builder = graph_builder
        self.neighbor_retriever = NeighborRetriever(self.graph_builder.graph)
        self.evidence_prop = EvidencePropagation(self.graph_builder.graph)
        self.is_ready = False

    def load(self) -> bool:
        hybrid_ready = self.hybrid_retriever.load()
        if hybrid_ready and self.graph_builder.graph.G.number_of_nodes() > 0:
            self.is_ready = True
            logger.info(f"  ✅ Graph-Aware Retriever listo (Nodos={self.graph_builder.graph.G.number_of_nodes()})")
        return self.is_ready

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        propagation_iterations: int = 2
    ) -> list[dict]:
        """Búsqueda consciente del grafo."""
        
        if not self.is_ready:
            return self.hybrid_retriever.search(query, top_k, filters=filters)

        # 1. Retrieval Inicial Híbrido (sobre investigadores)
        initial_results = self.hybrid_retriever.search(query, top_k=top_k*3, filters=filters, include_explanation=True)
        
        # 2. Preparar initial scores para propagación
        initial_scores = {}
        profile_map = {}
        if initial_results:
            for r in initial_results:
                pid = f"snii:{r['id']}"
                initial_scores[pid] = r["score"]
                profile_map[pid] = r
                
        # Buscar coincidencias léxicas simples en otros nodos del grafo (Topics, Papers)
        query_lower = query.lower()
        for nid, data in self.graph_builder.graph.G.nodes(data=True):
            ntype = data.get("type")
            text_to_match = data.get("label", "") or data.get("title", "")
            if text_to_match and query_lower in text_to_match.lower():
                # Dar un score inicial a estos nodos para que propaguen hacia sus autores
                initial_scores[nid] = initial_scores.get(nid, 0) + 0.8
                
        if not initial_scores:
            return []

        # 3. Propagar Evidencia
        refined_scores, provenance = self.evidence_prop.propagate(
            initial_scores, 
            iterations=propagation_iterations,
            decay_factor=0.3
        )

        # 4. Consolidar Resultados (Devolver investigadores, papers, topicos, etc)
        final_results = []
        
        for node_id, score in refined_scores.items():
            ntype = self.graph_builder.graph.G.nodes[node_id].get("type")
            
            if node_id.startswith("snii:"):
                orig_id = node_id.split("snii:")[1]
                
                if node_id in profile_map:
                    r = profile_map[node_id]
                    old_score = r["score"]
                    r["score"] = score
                    
                    if score > old_score + 0.01:
                        provs = provenance.get(node_id, [])
                        if provs:
                            sources = set(p[1].replace("_", " ") for p in provs)
                            r["explanation"] += f" | 🌐 Impulsado por red: {', '.join(sources)[:100]}"
                    
                    r["search_method"] = "graph_aware"
                    r["node_type"] = "researcher"
                    final_results.append(r)
                else:
                    full_profile = self.hybrid_retriever._profiles_by_id.get(orig_id)
                    if full_profile:
                        provs = provenance.get(node_id, [])
                        sources = set(p[1].replace("_", " ") for p in provs)
                        explanation = f"🌐 Conectado vía: {', '.join(sources)[:150]}"
                        
                        final_results.append({
                            "id": orig_id,
                            "nombre_completo": full_profile.get("nombre_completo", ""),
                            "full_name": full_profile.get("nombre_completo", ""),
                            "institucion": full_profile.get("institucion", ""),
                            "area": full_profile.get("area", ""),
                            "disciplina": full_profile.get("disciplina", ""),
                            "score": score,
                            "confidence": score,
                            "search_method": "graph_aware",
                            "node_type": "researcher",
                            "explanation": explanation
                        })
            elif ntype in [NodeType.PAPER.value, NodeType.TOPIC.value, NodeType.INSTITUTION.value]:
                # Devolver estos nodos tambien!
                data = self.graph_builder.graph.G.nodes[node_id]
                title = data.get("title") or data.get("label") or node_id
                
                provs = provenance.get(node_id, [])
                sources = set(p[1].replace("_", " ") for p in provs if p[1] != node_id)
                explanation = f"Relevante por propagación"
                if sources:
                    explanation += f" desde: {', '.join(sources)[:150]}"
                elif query_lower in title.lower():
                    explanation = "Coincidencia directa con la búsqueda"
                    
                final_results.append({
                    "id": node_id,
                    "nombre_completo": title,
                    "full_name": title,
                    "institucion": f"Type: {ntype.capitalize()}",
                    "area": "",
                    "disciplina": "",
                    "score": score,
                    "confidence": score,
                    "search_method": "graph_aware",
                    "node_type": ntype,
                    "explanation": explanation
                })

        # 5. Ordenar, filtrar y aplicar metadata filters si es nuevo
        if filters:
            # filters apply mostly to researchers
            filtered_results = []
            for r in final_results:
                if r.get("node_type") != "researcher":
                    filtered_results.append(r)
                    continue
                # For researchers, apply logic
                if "institution" in filters and filters["institution"].lower() not in str(r.get("institucion", "")).lower():
                    continue
                filtered_results.append(r)
            final_results = filtered_results
            
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results[:top_k]

    def stats(self) -> dict:
        return {
            "search_method": "graph_aware",
            "nodes": self.graph_builder.graph.G.number_of_nodes(),
            "edges": self.graph_builder.graph.G.number_of_edges(),
            "hybrid_ready": self.hybrid_retriever._get_search_method() != "none"
        }
