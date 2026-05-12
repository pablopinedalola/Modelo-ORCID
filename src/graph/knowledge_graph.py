"""
knowledge_graph.py -- Grafo academico para desambiguacion de identidad.

Modela las relaciones entre investigadores, candidatos, publicaciones,
instituciones y disciplinas como un grafo dirigido (NetworkX DiGraph).

El grafo es la estructura central sobre la que opera el modelo matematico
de refinamiento iterativo:

    d_{n+1}(v) = F(d_n(v))
    F(d)(v) = d(v) V  V_{u in N(v)} phi(d(u))

donde N(v) son los vecinos del nodo v en este grafo.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any

import networkx as nx

from src.models.schemas import (
    NodeType,
    EdgeType,
    NormalizedRecord,
    Candidate,
)

logger = logging.getLogger(__name__)


class AcademicKnowledgeGraph:
    """Grafo academico construido con NetworkX.

    Nodos tipados:
        - RESEARCHER: investigador SNII
        - CANDIDATE: candidato ORCID/OpenAlex
        - PAPER: publicacion academica
        - INSTITUTION: institucion con ROR
        - DISCIPLINE: area/disciplina

    Aristas tipadas:
        - CANDIDATE_FOR: candidato -> investigador
        - AFFILIATED_WITH: candidato/investigador -> institucion
        - AUTHORED: candidato -> paper
        - COAUTHOR: candidato <-> candidato
        - BELONGS_TO_AREA: investigador/candidato -> disciplina
        - HAS_ORCID: candidato -> orcid_node

    Examples:
        >>> graph = AcademicKnowledgeGraph()
        >>> graph.add_researcher(normalized_record)
        >>> graph.add_candidate(candidate, researcher_id)
        >>> print(graph.stats())
    """

    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self._node_counter: dict[str, int] = {t.value: 0 for t in NodeType}

    # ==================================================================
    # AGREGAR NODOS
    # ==================================================================

    def add_researcher(self, record: NormalizedRecord) -> str:
        """Agrega un investigador SNII como nodo.

        Args:
            record: Registro normalizado.

        Returns:
            Node ID del investigador.
        """
        node_id = f"snii:{record.id}"
        self.G.add_node(
            node_id,
            type=NodeType.RESEARCHER.value,
            label=record.original.full_name,
            normalized_name=record.normalized_name,
            institution=record.normalized_institution,
            area=record.original.area,
            discipline=record.original.disciplina,
            nivel=record.original.nivel.value,
            ror_id=record.ror_id,
        )
        self._node_counter[NodeType.RESEARCHER.value] += 1

        # Agregar institucion y relacion
        if record.normalized_institution:
            inst_id = self.add_institution(
                record.normalized_institution,
                ror_id=record.ror_id,
            )
            self.G.add_edge(
                node_id, inst_id,
                type=EdgeType.AFFILIATED_WITH.value,
                source="snii",
            )

        # Agregar disciplina
        if record.original.disciplina:
            disc_id = self.add_discipline(
                record.original.disciplina,
                area=record.original.area,
            )
            self.G.add_edge(
                node_id, disc_id,
                type=EdgeType.BELONGS_TO_AREA.value,
            )

        logger.debug(f"  Graph: +researcher '{record.original.full_name}'")
        return node_id

    def add_candidate(
        self,
        candidate: Candidate,
        researcher_id: str,
    ) -> str:
        """Agrega un candidato y lo vincula al investigador SNII.

        Args:
            candidate: Candidato ORCID/OpenAlex.
            researcher_id: Node ID del investigador SNII.

        Returns:
            Node ID del candidato.
        """
        cand_id = f"cand:{candidate.source.value}:{candidate.source_id}"

        # Evitar duplicados
        if cand_id in self.G:
            # Solo agregar la arista candidate_for si no existe
            if not self.G.has_edge(cand_id, researcher_id):
                self.G.add_edge(
                    cand_id, researcher_id,
                    type=EdgeType.CANDIDATE_FOR.value,
                )
            return cand_id

        self.G.add_node(
            cand_id,
            type=NodeType.CANDIDATE.value,
            label=candidate.display_name,
            source=candidate.source.value,
            orcid_id=candidate.orcid_id,
            openalex_id=candidate.openalex_id,
            works_count=candidate.works_count,
            cited_by_count=candidate.cited_by_count,
            concepts=candidate.concepts[:10],
        )
        self._node_counter[NodeType.CANDIDATE.value] += 1

        # Relacion: candidato -> investigador
        self.G.add_edge(
            cand_id, researcher_id,
            type=EdgeType.CANDIDATE_FOR.value,
        )

        # Agregar afiliaciones del candidato
        for aff in candidate.affiliations:
            inst_id = self.add_institution(aff)
            self.G.add_edge(
                cand_id, inst_id,
                type=EdgeType.AFFILIATED_WITH.value,
                source=candidate.source.value,
            )

        # Agregar conceptos como disciplinas
        for concept in candidate.concepts[:5]:
            disc_id = self.add_discipline(concept)
            self.G.add_edge(
                cand_id, disc_id,
                type=EdgeType.BELONGS_TO_AREA.value,
            )

        return cand_id

    def add_paper(
        self,
        paper_data: dict,
        author_node_id: str,
    ) -> str:
        """Agrega una publicacion y la vincula a su autor.

        Args:
            paper_data: Dict con id, title, publication_year, etc.
            author_node_id: Node ID del autor (candidato).

        Returns:
            Node ID del paper.
        """
        paper_id_raw = paper_data.get("id", "")
        if "openalex.org" in paper_id_raw:
            short = paper_id_raw.split("/")[-1]
        else:
            short = paper_id_raw or f"paper_{self._node_counter['paper']}"

        node_id = f"paper:{short}"

        if node_id not in self.G:
            self.G.add_node(
                node_id,
                type=NodeType.PAPER.value,
                label=paper_data.get("title", "")[:80],
                title=paper_data.get("title", ""),
                year=paper_data.get("publication_year"),
                cited_by=paper_data.get("cited_by_count", 0),
                doi=paper_data.get("doi", ""),
            )
            self._node_counter[NodeType.PAPER.value] += 1

        # Relacion: autor -> paper
        if not self.G.has_edge(author_node_id, node_id):
            self.G.add_edge(
                author_node_id, node_id,
                type=EdgeType.AUTHORED.value,
            )

        # Extraer coautores del paper (si disponibles)
        authorships = paper_data.get("authorships", [])
        for auth in authorships:
            if not isinstance(auth, dict):
                continue
            author_obj = auth.get("author", {})
            if not isinstance(author_obj, dict):
                continue
            coauthor_oalex = author_obj.get("id", "")
            if coauthor_oalex and coauthor_oalex != author_node_id:
                coauthor_name = author_obj.get("display_name", "unknown")
                co_id = f"cand:openalex:{coauthor_oalex.split('/')[-1]}"
                if co_id not in self.G:
                    self.G.add_node(
                        co_id,
                        type=NodeType.CANDIDATE.value,
                        label=coauthor_name,
                        source="openalex_coauthor",
                    )
                if not self.G.has_edge(author_node_id, co_id):
                    self.G.add_edge(
                        author_node_id, co_id,
                        type=EdgeType.COAUTHOR.value,
                    )

        return node_id

    def add_institution(
        self,
        name: str,
        ror_id: Optional[str] = None,
    ) -> str:
        """Agrega una institucion al grafo.

        Args:
            name: Nombre de la institucion.
            ror_id: ROR ID opcional.

        Returns:
            Node ID de la institucion.
        """
        # Usar ROR ID como clave si disponible, sino el nombre normalizado
        key = ror_id or name.strip().lower().replace(" ", "_")[:50]
        node_id = f"inst:{key}"

        if node_id not in self.G:
            self.G.add_node(
                node_id,
                type=NodeType.INSTITUTION.value,
                label=name,
                ror_id=ror_id,
            )
            self._node_counter[NodeType.INSTITUTION.value] += 1

        return node_id

    def add_discipline(
        self,
        name: str,
        area: str = "",
    ) -> str:
        """Agrega una disciplina al grafo.

        Args:
            name: Nombre de la disciplina.
            area: Area CONAHCyT (I-VII) opcional.

        Returns:
            Node ID de la disciplina.
        """
        key = name.strip().lower().replace(" ", "_")[:50]
        node_id = f"disc:{key}"

        if node_id not in self.G:
            self.G.add_node(
                node_id,
                type=NodeType.DISCIPLINE.value,
                label=name,
                area=area,
            )
            self._node_counter[NodeType.DISCIPLINE.value] += 1

        return node_id

    def add_coauthor_relation(self, node_a: str, node_b: str) -> None:
        """Agrega relacion de coautoria bidireccional."""
        if not self.G.has_edge(node_a, node_b):
            self.G.add_edge(node_a, node_b, type=EdgeType.COAUTHOR.value)
        if not self.G.has_edge(node_b, node_a):
            self.G.add_edge(node_b, node_a, type=EdgeType.COAUTHOR.value)

    # ==================================================================
    # CONSULTAS
    # ==================================================================

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
    ) -> list[dict]:
        """Obtiene vecinos de un nodo con su metadata.

        N(v) en el modelo matematico.

        Args:
            node_id: ID del nodo central.
            edge_type: Filtrar por tipo de arista (opcional).

        Returns:
            Lista de dicts {node_id, node_data, edge_data}.
        """
        if node_id not in self.G:
            return []

        neighbors = []

        # Vecinos salientes (successors)
        for succ in self.G.successors(node_id):
            edge_data = self.G.edges[node_id, succ]
            if edge_type and edge_data.get("type") != edge_type.value:
                continue
            neighbors.append({
                "node_id": succ,
                "node_data": dict(self.G.nodes[succ]),
                "edge_type": edge_data.get("type", ""),
            })

        # Vecinos entrantes (predecessors)
        for pred in self.G.predecessors(node_id):
            edge_data = self.G.edges[pred, node_id]
            if edge_type and edge_data.get("type") != edge_type.value:
                continue
            neighbors.append({
                "node_id": pred,
                "node_data": dict(self.G.nodes[pred]),
                "edge_type": edge_data.get("type", ""),
            })

        return neighbors

    def get_candidates_for(self, researcher_id: str) -> list[str]:
        """Obtiene IDs de candidatos vinculados a un investigador."""
        candidates = []
        for pred in self.G.predecessors(researcher_id):
            edge = self.G.edges[pred, researcher_id]
            if edge.get("type") == EdgeType.CANDIDATE_FOR.value:
                candidates.append(pred)
        return candidates

    def get_node_type(self, node_id: str) -> Optional[str]:
        """Retorna el tipo de un nodo."""
        if node_id in self.G:
            return self.G.nodes[node_id].get("type")
        return None

    def get_subgraph(
        self,
        center_node: str,
        depth: int = 2,
    ) -> nx.DiGraph:
        """Extrae subgrafo de profundidad limitada alrededor de un nodo.

        Args:
            center_node: Nodo central.
            depth: Profundidad maxima de expansion.

        Returns:
            Subgrafo como DiGraph.
        """
        if center_node not in self.G:
            return nx.DiGraph()

        # BFS para encontrar nodos dentro de la profundidad
        visited = {center_node}
        frontier = {center_node}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for succ in self.G.successors(node):
                    if succ not in visited:
                        visited.add(succ)
                        next_frontier.add(succ)
                for pred in self.G.predecessors(node):
                    if pred not in visited:
                        visited.add(pred)
                        next_frontier.add(pred)
            frontier = next_frontier

        return self.G.subgraph(visited).copy()

    # ==================================================================
    # EXPORTACION
    # ==================================================================

    def export_json(
        self,
        center_node: Optional[str] = None,
        depth: int = 2,
    ) -> dict:
        """Exporta el grafo (o subgrafo) en formato JSON para D3.js/vis.js.

        Args:
            center_node: Si se especifica, exporta solo subgrafo.
            depth: Profundidad para subgrafo.

        Returns:
            Dict con "nodes" y "edges" listos para visualizacion.
        """
        if center_node:
            G = self.get_subgraph(center_node, depth)
        else:
            G = self.G

        # Colores por tipo de nodo
        color_map = {
            NodeType.RESEARCHER.value: "#2196F3",   # Azul
            NodeType.CANDIDATE.value: "#FF9800",     # Naranja
            NodeType.PAPER.value: "#4CAF50",         # Verde
            NodeType.INSTITUTION.value: "#9C27B0",   # Morado
            NodeType.DISCIPLINE.value: "#F44336",    # Rojo
            NodeType.ORCID.value: "#A5D6A7",         # Verde claro
        }

        nodes = []
        for nid, data in G.nodes(data=True):
            ntype = data.get("type", "unknown")
            nodes.append({
                "id": nid,
                "label": data.get("label", nid)[:50],
                "type": ntype,
                "color": color_map.get(ntype, "#999"),
                **{k: v for k, v in data.items()
                   if k not in ("type", "label") and isinstance(v, (str, int, float, bool))},
            })

        edges = []
        for src, tgt, data in G.edges(data=True):
            edges.append({
                "source": src,
                "target": tgt,
                "type": data.get("type", "unknown"),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }

    def save_json(self, filepath: str, **kwargs) -> None:
        """Guarda el grafo como JSON."""
        data = self.export_json(**kwargs)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"  Graph exported: {filepath}")

    # ==================================================================
    # ESTADISTICAS
    # ==================================================================

    def stats(self) -> dict:
        """Retorna estadisticas del grafo."""
        type_counts: dict[str, int] = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, data in self.G.edges(data=True):
            t = data.get("type", "unknown")
            edge_counts[t] = edge_counts.get(t, 0) + 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "nodes_by_type": type_counts,
            "edges_by_type": edge_counts,
        }
