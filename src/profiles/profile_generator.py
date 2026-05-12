"""
profile_generator.py -- Generador de perfiles academicos.

Genera perfiles enriquecidos tipo "Wikipedia/OpenAlex" para
investigadores SNII con identidad resuelta. Exporta a JSON,
Markdown y HTML estatico.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import OUTPUT_DIR
from src.models.schemas import (
    NormalizedRecord,
    Candidate,
    ResearcherProfile,
)

logger = logging.getLogger(__name__)


class ProfileGenerator:
    """Genera perfiles academicos enriquecidos.

    Combina datos del SNII, ORCID, OpenAlex y el refinamiento
    para crear perfiles navegables tipo Wikipedia.

    Examples:
        >>> gen = ProfileGenerator()
        >>> profile = gen.generate_profile(record, best_candidate)
        >>> gen.save_html(profile)
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or (OUTPUT_DIR / "profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_profile(
        self,
        record: NormalizedRecord,
        candidate: Optional[Candidate],
        confidence: float = 0.0,
        papers: Optional[list[dict]] = None,
        semantic_score: float = 0.0,
    ) -> ResearcherProfile:
        """Genera un perfil academico completo.

        Args:
            record: Investigador SNII normalizado.
            candidate: Mejor candidato (puede ser None).
            confidence: Confidence score final.
            papers: Lista de publicaciones del candidato.
            semantic_score: Score semantico.

        Returns:
            ResearcherProfile completo.
        """
        profile = ResearcherProfile(
            id=record.id,
            full_name=record.original.full_name,
            snii_level=record.original.nivel.value,
            institution=record.normalized_institution,
            department=record.original.dependencia,
            area=f"{record.original.area} - {record.original.area_label}",
            discipline=record.original.disciplina,
            ror_id=record.ror_id,
            confidence=confidence,
            validated=confidence > 0.5,
        )

        if candidate:
            profile.orcid_id = candidate.orcid_id
            profile.openalex_id = candidate.openalex_id
            profile.works_count = candidate.works_count
            profile.cited_by_count = candidate.cited_by_count
            profile.concepts = candidate.concepts[:15]

        if papers:
            profile.publications = papers[:20]

        # External links
        links = {}
        if profile.orcid_id:
            links["ORCID"] = f"https://orcid.org/{profile.orcid_id}"
        if profile.openalex_id:
            oa_id = profile.openalex_id
            if not oa_id.startswith("http"):
                oa_id = f"https://openalex.org/authors/{oa_id}"
            links["OpenAlex"] = oa_id
        if profile.ror_id:
            links["ROR"] = profile.ror_id
        profile.external_links = links

        return profile

    def to_json(self, profile: ResearcherProfile) -> str:
        """Serializa perfil a JSON."""
        return profile.to_json()

    def to_markdown(self, profile: ResearcherProfile) -> str:
        """Genera vista Markdown del perfil."""
        lines = []
        lines.append(f"# {profile.full_name}")
        lines.append("")
        lines.append(f"**Nivel SNII:** {profile.snii_level}")
        lines.append(f"**Institucion:** {profile.institution}")
        if profile.department:
            lines.append(f"**Departamento:** {profile.department}")
        lines.append(f"**Area:** {profile.area}")
        lines.append(f"**Disciplina:** {profile.discipline}")
        lines.append("")

        # Identifiers
        lines.append("## Identificadores")
        if profile.orcid_id:
            lines.append(f"- **ORCID:** [{profile.orcid_id}](https://orcid.org/{profile.orcid_id})")
        if profile.openalex_id:
            lines.append(f"- **OpenAlex:** {profile.openalex_id}")
        if profile.ror_id:
            lines.append(f"- **ROR:** {profile.ror_id}")
        lines.append(f"- **Confidence:** {profile.confidence:.3f}")
        lines.append(f"- **Validado:** {'Si' if profile.validated else 'No'}")
        lines.append("")

        # Metrics
        if profile.works_count or profile.cited_by_count:
            lines.append("## Metricas")
            lines.append(f"- Publicaciones: {profile.works_count}")
            lines.append(f"- Citas: {profile.cited_by_count}")
            lines.append("")

        # Concepts
        if profile.concepts:
            lines.append("## Temas de Investigacion")
            for c in profile.concepts[:10]:
                lines.append(f"- {c}")
            lines.append("")

        # Publications
        if profile.publications:
            lines.append("## Publicaciones Destacadas")
            for p in profile.publications[:10]:
                title = p.get("title", "Sin titulo")
                year = p.get("publication_year", "")
                cited = p.get("cited_by_count", 0)
                lines.append(f"- **{title}** ({year}) - {cited} citas")
            lines.append("")

        lines.append(f"---")
        lines.append(f"*Generado por Modelo-ORCID el {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

        return "\n".join(lines)

    def to_html(self, profile: ResearcherProfile) -> str:
        """Genera pagina HTML tipo Wikipedia/OpenAlex."""

        # Confidence bar color
        conf = profile.confidence
        if conf >= 0.7:
            conf_color = "#4CAF50"
            conf_label = "Alta"
        elif conf >= 0.4:
            conf_color = "#FF9800"
            conf_label = "Media"
        else:
            conf_color = "#f44336"
            conf_label = "Baja"

        conf_pct = int(conf * 100)

        # Build concepts HTML
        concepts_html = ""
        if profile.concepts:
            tags = []
            for c in profile.concepts[:12]:
                tags.append(f'<span class="tag">{c}</span>')
            concepts_html = f"""
            <section class="card">
                <h2>Temas de Investigacion</h2>
                <div class="tags">{''.join(tags)}</div>
            </section>"""

        # Build publications HTML
        pubs_html = ""
        if profile.publications:
            rows = []
            for p in profile.publications[:10]:
                title = p.get("title", "Sin titulo") or "Sin titulo"
                year = p.get("publication_year", "")
                cited = p.get("cited_by_count", 0)
                doi = p.get("doi", "")
                doi_link = f'<a href="{doi}" target="_blank">DOI</a>' if doi else ""
                rows.append(f"""
                <tr>
                    <td class="pub-title">{title}</td>
                    <td>{year}</td>
                    <td>{cited}</td>
                    <td>{doi_link}</td>
                </tr>""")
            pubs_html = f"""
            <section class="card">
                <h2>Publicaciones Destacadas</h2>
                <table class="pub-table">
                    <thead>
                        <tr><th>Titulo</th><th>Ano</th><th>Citas</th><th>Link</th></tr>
                    </thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </section>"""

        # Build identifiers
        ids_html = ""
        if profile.orcid_id:
            ids_html += f'''
            <div class="id-item">
                <span class="id-label">ORCID</span>
                <a href="https://orcid.org/{profile.orcid_id}" target="_blank">{profile.orcid_id}</a>
            </div>'''
        if profile.openalex_id:
            oa_url = profile.openalex_id if profile.openalex_id.startswith("http") else f"https://openalex.org/authors/{profile.openalex_id}"
            ids_html += f'''
            <div class="id-item">
                <span class="id-label">OpenAlex</span>
                <a href="{oa_url}" target="_blank">{profile.openalex_id.split("/")[-1]}</a>
            </div>'''
        if profile.ror_id:
            ids_html += f'''
            <div class="id-item">
                <span class="id-label">ROR</span>
                <a href="{profile.ror_id}" target="_blank">{profile.ror_id.split("/")[-1]}</a>
            </div>'''

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{profile.full_name} - Modelo-ORCID</title>
    <meta name="description" content="Perfil academico de {profile.full_name}, investigador SNII nivel {profile.snii_level}">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f1117;
            --surface: #1a1d27;
            --surface2: #232636;
            --border: #2e3244;
            --text: #e4e6f0;
            --text-dim: #8a8ea3;
            --accent: #6366f1;
            --accent-glow: rgba(99,102,241,0.15);
            --green: #4ade80;
            --orange: #fb923c;
            --red: #f87171;
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 40px 20px; }}
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .badge-snii {{ background: var(--accent-glow); color: var(--accent); border: 1px solid var(--accent); }}
        .badge-validated {{ background: rgba(74,222,128,0.12); color: var(--green); border: 1px solid rgba(74,222,128,0.3); }}
        .badge-unvalidated {{ background: rgba(248,113,113,0.12); color: var(--red); border: 1px solid rgba(248,113,113,0.3); }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 36px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .subtitle {{
            color: var(--text-dim);
            font-size: 15px;
            margin-bottom: 16px;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 20px;
        }}
        .meta-item {{ font-size: 14px; }}
        .meta-item .label {{ color: var(--text-dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .meta-item .value {{ font-weight: 500; margin-top: 2px; }}

        /* Cards */
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 20px;
        }}
        .card h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--accent);
        }}

        /* Confidence */
        .confidence-bar {{
            background: var(--surface2);
            border-radius: 8px;
            height: 28px;
            overflow: hidden;
            position: relative;
            margin: 8px 0;
        }}
        .confidence-fill {{
            height: 100%;
            border-radius: 8px;
            transition: width 1s ease;
            display: flex;
            align-items: center;
            padding-left: 12px;
            font-size: 13px;
            font-weight: 600;
            color: #fff;
        }}

        /* Metrics */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        .metric {{
            text-align: center;
            padding: 16px;
            background: var(--surface2);
            border-radius: 10px;
        }}
        .metric .num {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
        .metric .label {{ font-size: 12px; color: var(--text-dim); margin-top: 4px; }}

        /* IDs */
        .id-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid var(--border);
        }}
        .id-item:last-child {{ border-bottom: none; }}
        .id-label {{
            background: var(--accent-glow);
            color: var(--accent);
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            min-width: 70px;
            text-align: center;
        }}
        .id-item a {{ color: var(--text); text-decoration: none; }}
        .id-item a:hover {{ color: var(--accent); }}

        /* Tags */
        .tags {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .tag {{
            background: var(--surface2);
            border: 1px solid var(--border);
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 13px;
            color: var(--text-dim);
        }}

        /* Publications table */
        .pub-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        .pub-table th {{
            text-align: left;
            padding: 10px 8px;
            border-bottom: 2px solid var(--border);
            color: var(--text-dim);
            font-size: 12px;
            text-transform: uppercase;
        }}
        .pub-table td {{
            padding: 10px 8px;
            border-bottom: 1px solid var(--border);
        }}
        .pub-title {{ max-width: 400px; }}
        .pub-table a {{ color: var(--accent); text-decoration: none; }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 32px;
            color: var(--text-dim);
            font-size: 12px;
        }}
        .footer a {{ color: var(--accent); text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div style="display:flex; gap:8px; margin-bottom:12px;">
                <span class="badge badge-snii">SNII Nivel {profile.snii_level}</span>
                <span class="badge {'badge-validated' if profile.validated else 'badge-unvalidated'}">
                    {'Identidad Verificada' if profile.validated else 'Identidad Pendiente'}
                </span>
            </div>
            <h1>{profile.full_name}</h1>
            <div class="subtitle">{profile.institution} &mdash; {profile.discipline}</div>
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="label">Area</div>
                    <div class="value">{profile.area}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Departamento</div>
                    <div class="value">{profile.department or 'No especificado'}</div>
                </div>
            </div>
        </div>

        <!-- Confidence -->
        <div class="card">
            <h2>Confidence Score</h2>
            <p style="color:var(--text-dim); font-size:14px; margin-bottom:8px;">
                Nivel de confianza en la resolucion de identidad: <strong style="color:{conf_color}">{conf_label} ({conf_pct}%)</strong>
            </p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{conf_pct}%; background:linear-gradient(90deg, {conf_color}88, {conf_color});">
                    {conf_pct}%
                </div>
            </div>
        </div>

        <!-- Metrics -->
        <div class="card">
            <h2>Metricas</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="num">{profile.works_count}</div>
                    <div class="label">Publicaciones</div>
                </div>
                <div class="metric">
                    <div class="num">{profile.cited_by_count}</div>
                    <div class="label">Citas</div>
                </div>
                <div class="metric">
                    <div class="num">{profile.h_index}</div>
                    <div class="label">H-Index</div>
                </div>
            </div>
        </div>

        <!-- Identifiers -->
        <div class="card">
            <h2>Identificadores</h2>
            {ids_html if ids_html else '<p style="color:var(--text-dim)">Sin identificadores externos resueltos</p>'}
        </div>

        <!-- Concepts -->
        {concepts_html}

        <!-- Publications -->
        {pubs_html}

        <!-- Footer -->
        <div class="footer">
            <p>Generado por <a href="#">Modelo-ORCID</a> &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p>Sistema de Desambiguacion de Identidad Academica para Investigadores SNII</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def save_profile(
        self,
        profile: ResearcherProfile,
        formats: tuple[str, ...] = ("html", "json"),
    ) -> list[Path]:
        """Guarda perfil en multiples formatos.

        Args:
            profile: Perfil a guardar.
            formats: Formatos de salida ("html", "json", "md").

        Returns:
            Lista de paths creados.
        """
        slug = profile.full_name.lower().replace(" ", "_")[:40]
        paths = []

        if "json" in formats:
            p = self.output_dir / f"{slug}.json"
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.to_json(profile))
            paths.append(p)

        if "md" in formats:
            p = self.output_dir / f"{slug}.md"
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.to_markdown(profile))
            paths.append(p)

        if "html" in formats:
            p = self.output_dir / f"{slug}.html"
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.to_html(profile))
            paths.append(p)

        logger.info(f"  Profile saved: {slug} ({', '.join(formats)})")
        return paths
