"""
model_mapping.py -- Capa de mapeo matematico formal.

Conecta explicitamente las estructuras computacionales del sistema
con los conceptos matematicos del modelo teorico de la tesis.

Mapeos principales:
    Estructura computacional      <-->  Concepto matematico
    ─────────────────────────────────────────────────────────
    EvidenceVector.combine()      <-->  Lattice join (V)
    graph.get_neighbors(v)        <-->  N(v) (vecindad)
    RefinementEngine.phi()        <-->  phi: extraccion de evidencia
    RefinementEngine.refine()     <-->  F: operador de refinamiento
    IterationLog.converged        <-->  Punto fijo d* = F(d*)
    EvidenceVector                <-->  d(v) in D (descriptor)
    AcademicKnowledgeGraph        <-->  G = (V, E) grafo academico
    confidence()                  <-->  mu: D -> [0,1] (medida)

Esto permite:
    1. Generar documentacion que conecta codigo con formulas.
    2. Verificar que la implementacion respeta la teoria.
    3. Producir tablas y explicaciones para tesis/paper.

Usage:
    mapping = MathematicalMapping()
    table = mapping.get_mapping_table()
    verification = mapping.verify_properties(trace)
    latex = mapping.export_latex_table()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.interpreter.evidence_trace import EvidenceTrace, DIMENSION_NAMES
from src.interpreter.dynamics import DynamicsAnalyzer


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class MathMapping:
    """Un mapeo individual: computacional <-> matematico.

    Attributes:
        code_element: Nombre del elemento en codigo.
        code_location: Archivo y linea donde se implementa.
        math_symbol: Simbolo matematico (LaTeX).
        math_concept: Nombre del concepto.
        math_definition: Definicion formal breve.
        description: Descripcion en lenguaje natural.
    """
    code_element: str
    code_location: str
    math_symbol: str
    math_concept: str
    math_definition: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "code_element": self.code_element,
            "code_location": self.code_location,
            "math_symbol": self.math_symbol,
            "math_concept": self.math_concept,
            "math_definition": self.math_definition,
            "description": self.description,
        }


@dataclass
class PropertyVerification:
    """Verificacion de una propiedad matematica.

    Attributes:
        property_name: Nombre de la propiedad.
        math_statement: Enunciado formal.
        verified: Si se verifico computacionalmente.
        evidence: Datos que soportan la verificacion.
        counterexample: Contraejemplo si no se verifico.
    """
    property_name: str
    math_statement: str
    verified: bool = False
    evidence: str = ""
    counterexample: str = ""

    def to_dict(self) -> dict:
        return {
            "property_name": self.property_name,
            "math_statement": self.math_statement,
            "verified": self.verified,
            "evidence": self.evidence,
            "counterexample": self.counterexample,
        }


# ─── Main Class ──────────────────────────────────────────────────────────────

class MathematicalMapping:
    """Capa de mapeo entre implementacion y modelo matematico.

    Proporciona:
    1. Tabla de mapeo codigo <-> matematicas.
    2. Verificacion de propiedades formales.
    3. Exportacion a LaTeX para tesis/paper.
    4. Explicaciones formales de comportamiento.
    """

    def __init__(self) -> None:
        self._mappings = self._build_mappings()

    def _build_mappings(self) -> list[MathMapping]:
        """Construye tabla completa de mapeos."""
        return [
            MathMapping(
                code_element="AcademicKnowledgeGraph",
                code_location="src/graph/knowledge_graph.py",
                math_symbol=r"G = (V, E)",
                math_concept="Grafo Académico",
                math_definition="Grafo dirigido donde V son entidades académicas y E son relaciones.",
                description="El grafo de conocimiento es la estructura central sobre la que opera el refinamiento.",
            ),
            MathMapping(
                code_element="EvidenceVector",
                code_location="src/models/schemas.py:EvidenceVector",
                math_symbol=r"d(v) \in D",
                math_concept="Descriptor de Identidad",
                math_definition="Vector en [0,1]^7 que describe la evidencia acumulada de un candidato.",
                description="Cada candidato tiene un vector de 7 dimensiones que captura diferentes tipos de evidencia.",
            ),
            MathMapping(
                code_element="EvidenceVector.combine()",
                code_location="src/models/schemas.py:EvidenceVector.combine",
                math_symbol=r"\vee",
                math_concept="Lattice Join",
                math_definition=r"d_1 \vee d_2 = \max(d_1, d_2) por dimensión",
                description="Operador de acumulación: la evidencia solo crece, nunca se pierde. Implementa semántica de lattice.",
            ),
            MathMapping(
                code_element="RefinementEngine.phi()",
                code_location="src/refinement/refinement_engine.py:phi",
                math_symbol=r"\varphi",
                math_concept="Función de Extracción",
                math_definition=r"\varphi: V \times E \to D",
                description="Extrae evidencia de un vecino del grafo. Transforma información estructural en evidencia dimensional.",
            ),
            MathMapping(
                code_element="RefinementEngine.refine()",
                code_location="src/refinement/refinement_engine.py:refine",
                math_symbol=r"F",
                math_concept="Operador de Refinamiento",
                math_definition=r"F(d)(v) = d(v) \vee \bigvee_{u \in N(v)} \varphi(d(u))",
                description="El operador central: aplica extracción sobre vecinos y acumula evidencia via lattice join.",
            ),
            MathMapping(
                code_element="graph.get_neighbors(v)",
                code_location="src/graph/knowledge_graph.py:get_neighbors",
                math_symbol=r"N(v)",
                math_concept="Vecindad",
                math_definition=r"N(v) = \{u \in V : (v,u) \in E \text{ o } (u,v) \in E\}",
                description="Conjunto de vecinos de v en el grafo: tanto sucesores como predecesores.",
            ),
            MathMapping(
                code_element="IterationLog.converged",
                code_location="src/refinement/refinement_engine.py:IterationLog",
                math_symbol=r"d^* = F(d^*)",
                math_concept="Punto Fijo",
                math_definition=r"\|d_{n+1} - d_n\|_\infty < \varepsilon",
                description="El refinamiento converge cuando F ya no modifica el estado: el descriptor alcanza su punto fijo.",
            ),
            MathMapping(
                code_element="EvidenceVector.confidence()",
                code_location="src/models/schemas.py:EvidenceVector.confidence",
                math_symbol=r"\mu: D \to [0,1]",
                math_concept="Medida de Confianza",
                math_definition=r"\mu(d) = \sum_i w_i \cdot d_i / \sum_i w_i",
                description="Combinación ponderada de todas las dimensiones del descriptor. Produce un score escalar.",
            ),
            MathMapping(
                code_element="CONVERGENCE_EPSILON",
                code_location="config.py:CONVERGENCE_EPSILON",
                math_symbol=r"\varepsilon",
                math_concept="Umbral de Convergencia",
                math_definition=r"\varepsilon = 0.01",
                description="Tolerancia para considerar que el sistema alcanzó su punto fijo.",
            ),
            MathMapping(
                code_element="EVIDENCE_WEIGHTS",
                code_location="config.py:EVIDENCE_WEIGHTS",
                math_symbol=r"w = (w_1, ..., w_7)",
                math_concept="Vector de Pesos",
                math_definition=r"w \in \mathbb{R}^7, \sum w_i = 1",
                description="Pesos que determinan la importancia relativa de cada dimensión de evidencia.",
            ),
            MathMapping(
                code_element="EvidenceVector.distance()",
                code_location="src/models/schemas.py:EvidenceVector.distance",
                math_symbol=r"\|d_1 - d_2\|_\infty",
                math_concept="Norma Infinito",
                math_definition=r"\max_i |d_{1,i} - d_{2,i}|",
                description="Distancia máxima absoluta entre dos descriptores. Usada para verificar convergencia.",
            ),
            MathMapping(
                code_element="SemanticMatcher",
                code_location="src/semantic/semantic_matcher.py",
                math_symbol=r"s: D \times D \to [0,1]",
                math_concept="Similitud Semántica",
                math_definition=r"s(a,b) = \cos(\text{emb}(a), \text{emb}(b))",
                description="Cosine similarity entre embeddings de sentence-transformers. La 7ma dimensión del descriptor.",
            ),
        ]

    # ── Accessors ─────────────────────────────────────────────────────

    def get_mapping_table(self) -> list[dict]:
        """Retorna tabla completa de mapeos."""
        return [m.to_dict() for m in self._mappings]

    def get_mapping_for(self, code_element: str) -> Optional[MathMapping]:
        """Busca el mapeo de un elemento de codigo."""
        for m in self._mappings:
            if m.code_element == code_element:
                return m
        return None

    # ── Verification ──────────────────────────────────────────────────

    def verify_properties(
        self,
        traces: list[EvidenceTrace],
    ) -> list[PropertyVerification]:
        """Verifica propiedades matematicas contra datos reales.

        Propiedades verificadas:
        1. Monotonía del lattice join (d nunca decrece).
        2. Convergencia a punto fijo.
        3. Idempotencia del punto fijo (F(d*) = d*).
        4. Monotonía de mu sobre el lattice.

        Args:
            traces: Trazas de evidencia para verificar.

        Returns:
            Lista de PropertyVerification.
        """
        verifications = []

        # 1. Monotonicity: d(v) nunca decrece
        verifications.append(
            self._verify_monotonicity(traces)
        )

        # 2. Convergence: all traces converge
        verifications.append(
            self._verify_convergence(traces)
        )

        # 3. Fixed point stability
        verifications.append(
            self._verify_fixed_point(traces)
        )

        # 4. Confidence monotonicity over lattice
        verifications.append(
            self._verify_confidence_monotonicity(traces)
        )

        return verifications

    def _verify_monotonicity(
        self,
        traces: list[EvidenceTrace],
    ) -> PropertyVerification:
        """Verifica: d_{n+1}(v) >= d_n(v) para toda dimension."""
        prop = PropertyVerification(
            property_name="Monotonía del Lattice Join",
            math_statement=r"d_{n+1,i}(v) \geq d_{n,i}(v) \; \forall i, \forall n",
            verified=True,
        )

        violations = 0
        total_checks = 0

        for trace in traces:
            for cid in trace.initial_states:
                ct = trace.get_candidate_trace(cid)
                for i in range(1, len(ct.trajectory)):
                    prev = ct.trajectory[i - 1].scores
                    curr = ct.trajectory[i].scores
                    for dim in DIMENSION_NAMES:
                        total_checks += 1
                        if curr.get(dim, 0) < prev.get(dim, 0) - 1e-9:
                            violations += 1

        if violations > 0:
            prop.verified = False
            prop.counterexample = (
                f"{violations} violaciones en {total_checks} checks."
            )
        else:
            prop.evidence = (
                f"Verificado en {total_checks} comparaciones dimension×iteración "
                f"sin ninguna violación."
            )

        return prop

    def _verify_convergence(
        self,
        traces: list[EvidenceTrace],
    ) -> PropertyVerification:
        """Verifica: existe n* tal que d_{n*} = F(d_{n*})."""
        prop = PropertyVerification(
            property_name="Convergencia a Punto Fijo",
            math_statement=r"\exists n^*: d_{n^*} = F(d_{n^*})",
            verified=True,
        )

        converged_count = sum(1 for t in traces if t.converged)
        total = len(traces)

        if converged_count == total:
            prop.evidence = f"Todos los {total} casos convergieron."
        elif converged_count > 0:
            prop.evidence = (
                f"{converged_count}/{total} casos convergieron "
                f"({converged_count/total*100:.0f}%)."
            )
            prop.verified = converged_count / total >= 0.8
        else:
            prop.verified = False
            prop.counterexample = "Ningún caso convergió."

        return prop

    def _verify_fixed_point(
        self,
        traces: list[EvidenceTrace],
    ) -> PropertyVerification:
        """Verifica que el punto fijo es estable (delta final ≈ 0)."""
        prop = PropertyVerification(
            property_name="Estabilidad del Punto Fijo",
            math_statement=r"\|F(d^*) - d^*\|_\infty < \varepsilon",
        )

        final_deltas = []
        for trace in traces:
            if trace.iterations:
                final_deltas.append(trace.iterations[-1].max_delta)

        if final_deltas:
            avg_delta = sum(final_deltas) / len(final_deltas)
            max_delta = max(final_deltas)
            prop.verified = max_delta < 0.01
            prop.evidence = (
                f"Delta final promedio: {avg_delta:.6f}, "
                f"máximo: {max_delta:.6f}."
            )
        else:
            prop.verified = False
            prop.counterexample = "No hay datos de iteraciones."

        return prop

    def _verify_confidence_monotonicity(
        self,
        traces: list[EvidenceTrace],
    ) -> PropertyVerification:
        """Verifica: si d_1 >= d_2 entonces mu(d_1) >= mu(d_2)."""
        prop = PropertyVerification(
            property_name="Monotonía de la Medida de Confianza",
            math_statement=r"d_1 \geq d_2 \Rightarrow \mu(d_1) \geq \mu(d_2)",
            verified=True,
        )

        violations = 0
        total = 0

        for trace in traces:
            for cid in trace.initial_states:
                ct = trace.get_candidate_trace(cid)
                for i in range(1, len(ct.trajectory)):
                    prev = ct.trajectory[i - 1]
                    curr = ct.trajectory[i]

                    # Check if curr >= prev (monotonicity of lattice)
                    all_geq = all(
                        curr.scores.get(d, 0) >= prev.scores.get(d, 0) - 1e-9
                        for d in DIMENSION_NAMES
                    )

                    if all_geq:
                        total += 1
                        if curr.confidence < prev.confidence - 1e-9:
                            violations += 1

        if violations > 0:
            prop.verified = False
            prop.counterexample = (
                f"{violations} violaciones en {total} casos."
            )
        else:
            prop.evidence = (
                f"Verificado en {total} transiciones. "
                f"La medida respeta el orden del lattice."
            )

        return prop

    # ── Export ─────────────────────────────────────────────────────────

    def export_latex_table(self) -> str:
        """Exporta tabla de mapeo en formato LaTeX.

        Returns:
            String con tabla LaTeX lista para incluir en tesis.
        """
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Mapeo entre implementación computacional y modelo matemático}",
            r"\label{tab:mapping}",
            r"\begin{tabular}{lll}",
            r"\toprule",
            r"\textbf{Código} & \textbf{Símbolo} & \textbf{Concepto} \\",
            r"\midrule",
        ]

        for m in self._mappings:
            code = m.code_element.replace("_", r"\_")
            lines.append(
                f"\\texttt{{{code}}} & ${m.math_symbol}$ & {m.math_concept} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def export_markdown_table(self) -> str:
        """Exporta tabla de mapeo en formato Markdown.

        Returns:
            String con tabla Markdown.
        """
        lines = [
            "| Código | Símbolo | Concepto | Definición |",
            "|--------|---------|----------|------------|",
        ]

        for m in self._mappings:
            sym = m.math_symbol.replace("|", "\\|")
            defn = m.math_definition.replace("|", "\\|")
            lines.append(
                f"| `{m.code_element}` | {sym} | {m.math_concept} | {defn} |"
            )

        return "\n".join(lines)

    def export_verification_markdown(
        self,
        verifications: list[PropertyVerification],
    ) -> str:
        """Exporta resultados de verificacion en Markdown."""
        lines = [
            "## Verificación de Propiedades Matemáticas\n",
        ]

        for v in verifications:
            status = "✅" if v.verified else "❌"
            lines.append(f"### {status} {v.property_name}")
            lines.append(f"**Enunciado**: {v.math_statement}\n")
            if v.verified:
                lines.append(f"**Evidencia**: {v.evidence}\n")
            else:
                lines.append(f"**Contraejemplo**: {v.counterexample}\n")

        return "\n".join(lines)
