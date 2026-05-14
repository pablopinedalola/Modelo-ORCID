"""
citation_network.py — Construcción y análisis de la Red de Citas.

Convierte datos descargados de OpenAlex en un subgrafo orientado 
a medir impacto, influencia y descubrir redes de colaboración.
"""

from __future__ import annotations

import logging
import json
import networkx as nx
from pathlib import Path

from config import DATA_DIR
from src.models.schemas import NodeType, EdgeType
from src.graph.knowledge_graph import AcademicKnowledgeGraph

logger = logging.getLogger(__name__)

class CitationNetwork:
    """Modela la red de citas y tópicos."""

    def __init__(self, base_graph: AcademicKnowledgeGraph):
        self.graph = base_graph
        self.G = self.graph.G

    def build_from_openalex_cache(self, works_file: Path):
        """Integra papers descargados al knowledge graph."""
        if not works_file.exists():
            logger.warning(f"Archivo de works no encontrado: {works_file}")
            return
            
        with open(works_file) as f:
            works_data = json.load(f)

        logger.info(f"Construyendo red de citas a partir de {len(works_data)} perfiles de autor...")
        added_papers = 0
        added_citations = 0

        for author_id, works in works_data.items():
            # author_id es de openalex
            # Buscar el nodo investigador que mapea a esto, si aplica, aunque por ahora lo ligamos al paper.
            # Idealmente, creamos un nodo autor OpenAlex y vinculamos el paper.
            cand_id = f"cand:openalex:{author_id.split('/')[-1]}"
            if cand_id not in self.G:
                self.G.add_node(cand_id, type=NodeType.CANDIDATE.value, label=author_id)
            
            for w in works:
                if not w.get('id'): continue
                
                paper_id = f"paper:{w['id'].split('/')[-1]}"
                if paper_id not in self.G:
                    self.G.add_node(
                        paper_id,
                        type=NodeType.PAPER.value,
                        title=w.get("title", ""),
                        year=w.get("year"),
                        cited_by=w.get("cited_by_count", 0),
                        label=w.get("title", "")[:50]
                    )
                    added_papers += 1
                
                # Relacion Authored
                self.G.add_edge(cand_id, paper_id, type=EdgeType.AUTHORED.value)

                # Topics
                for topic in w.get("topics", []):
                    topic_name = topic.get("name")
                    if topic_name:
                        t_id = f"topic:{topic['id'].split('/')[-1]}"
                        if t_id not in self.G:
                            self.G.add_node(t_id, type=NodeType.TOPIC.value, label=topic_name)
                        self.G.add_edge(paper_id, t_id, type=EdgeType.RELATED_TO_TOPIC.value)

                # Referencias (Citations)
                for ref_id in w.get("referenced_works", []):
                    ref_short = f"paper:{ref_id.split('/')[-1]}"
                    if ref_short not in self.G:
                        self.G.add_node(ref_short, type=NodeType.PAPER.value, label="Unknown Referenced Paper")
                    self.G.add_edge(paper_id, ref_short, type=EdgeType.CITES.value)
                    added_citations += 1

        logger.info(f"Red de Citas Construida: +{added_papers} papers, +{added_citations} citas.")

    def get_influential_papers(self, top_k: int = 10) -> list[dict]:
        """Extrae papers más citados internamente en este grafo local."""
        in_degrees = [(n, d) for n, d in self.G.in_degree() if self.G.nodes[n].get("type") == NodeType.PAPER.value]
        # Filtrar por los que tienen título
        valid_papers = [(n, d) for n, d in in_degrees if self.G.nodes[n].get("title")]
        valid_papers.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for n, d in valid_papers[:top_k]:
            data = self.G.nodes[n]
            results.append({
                "id": n,
                "title": data.get("title", ""),
                "local_citations": d,
                "global_citations": data.get("cited_by", 0)
            })
        return results
