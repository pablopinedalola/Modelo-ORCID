"""
graph_enrichment.py — Construcción y Enriquecimiento del Grafo Académico.

Lee perfiles SNII procesados y construye un grafo rico en relaciones.
Inyecta aristas de similitud semántica, pertenencia a instituciones y áreas,
coautorías (si las hubiera), etc.

Permite que el retrieval opere sobre un Knowledge Graph.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from itertools import combinations

import networkx as nx

from config import PROCESSED_DATA_DIR, DATA_DIR
from src.graph.knowledge_graph import AcademicKnowledgeGraph
from src.models.schemas import NodeType

logger = logging.getLogger(__name__)

GRAPH_DIR = DATA_DIR / "graph"


class AcademicGraphBuilder:
    """Constructor y enriquecedor del Grafo Académico.
    
    Toma perfiles y construye relaciones de:
    - Afiliación (institution)
    - Área / Disciplina (area, discipline)
    - Similitud Semántica (semantic_similarity)
    """

    def __init__(self, store_dir: Path = GRAPH_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.graph = AcademicKnowledgeGraph()

    def build_from_profiles(self, profiles: list[dict]):
        """Construye el grafo base a partir de perfiles SNII."""
        logger.info("Construyendo grafo base...")
        
        # 1. Agregar nodos base (Investigadores, Instituciones, Disciplinas)
        for p in profiles:
            pid = f"snii:{p['id']}"
            
            # Investigador
            self.graph.G.add_node(
                pid,
                type=NodeType.RESEARCHER.value,
                label=p.get("nombre_completo", ""),
                institution=p.get("institucion", ""),
                area=p.get("area", ""),
                discipline=p.get("disciplina", ""),
                nivel=p.get("nivel", "")
            )
            
            # Institución
            inst = p.get("institucion", "").strip()
            if inst:
                inst_id = f"inst:{inst.lower().replace(' ', '_')[:50]}"
                if inst_id not in self.graph.G:
                    self.graph.G.add_node(inst_id, type=NodeType.INSTITUTION.value, label=inst)
                # Arista Investigador -> Institución
                self.graph.G.add_edge(pid, inst_id, type="affiliated_with", weight=1.0)
                # Arista Institución -> Investigador (para propagation bidireccional)
                self.graph.G.add_edge(inst_id, pid, type="has_researcher", weight=1.0)
                
            # Disciplina
            disc = p.get("disciplina", "").strip()
            area = p.get("area", "").strip()
            if disc:
                disc_id = f"disc:{disc.lower().replace(' ', '_')[:50]}"
                if disc_id not in self.graph.G:
                    self.graph.G.add_node(disc_id, type=NodeType.DISCIPLINE.value, label=disc, area=area)
                self.graph.G.add_edge(pid, disc_id, type="belongs_to_discipline", weight=1.0)
                self.graph.G.add_edge(disc_id, pid, type="has_researcher", weight=1.0)
                
            if area:
                area_id = f"area:{area}"
                if area_id not in self.graph.G:
                    self.graph.G.add_node(area_id, type="area", label=f"Area {area}")
                self.graph.G.add_edge(pid, area_id, type="belongs_to_area", weight=1.0)
                self.graph.G.add_edge(area_id, pid, type="has_researcher", weight=1.0)

    def enrich_semantic_similarity(self, embedding_pipeline, profiles: list[dict], threshold=0.75):
        """Calcula similitud semántica y agrega aristas similar_to."""
        logger.info("Enriqueciendo con similitud semántica...")
        if not embedding_pipeline or not embedding_pipeline.load_cache():
            logger.warning("No se pudo cargar cache semántico.")
            return

        # Computar embeddings para la lista de perfiles
        res = embedding_pipeline.generate_embeddings(profiles)
        embeddings = res["embeddings"]
        pids = res["profile_ids"]
        
        import numpy as np
        
        # Calcular pairwise cosine similarity
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embs = embeddings / np.where(norm == 0, 1e-10, norm)
        sim_matrix = np.dot(norm_embs, norm_embs.T)
        
        edges_added = 0
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    id1 = f"snii:{pids[i]}"
                    id2 = f"snii:{pids[j]}"
                    self.graph.G.add_edge(id1, id2, type="similar_to", weight=float(sim))
                    self.graph.G.add_edge(id2, id1, type="similar_to", weight=float(sim))
                    edges_added += 2
                    
        logger.info(f"Se agregaron {edges_added} aristas de similitud semántica (>= {threshold})")

    def save(self) -> Path:
        """Persiste el grafo completo."""
        graph_path = self.store_dir / "academic_graph.gpickle"
        self.graph.save_pickle(str(graph_path))
        
        meta_path = self.store_dir / "graph_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.graph.stats(), f, indent=2)
            
        logger.info(f"Grafo guardado exitosamente en {self.store_dir}")
        return graph_path

    @classmethod
    def load(cls, store_dir: Path = GRAPH_DIR) -> AcademicGraphBuilder:
        builder = cls(store_dir)
        graph_path = store_dir / "academic_graph.gpickle"
        if graph_path.exists():
            builder.graph = AcademicKnowledgeGraph.load_pickle(str(graph_path))
        return builder

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.rag.embedding_pipeline import EmbeddingPipeline
    
    profiles_path = PROCESSED_DATA_DIR / "snii_profiles.json"
    with open(profiles_path) as f:
        profiles = json.load(f)
        
    builder = AcademicGraphBuilder()
    builder.build_from_profiles(profiles)
    
    ep = EmbeddingPipeline()
    builder.enrich_semantic_similarity(ep, profiles, threshold=0.7)
    builder.save()
