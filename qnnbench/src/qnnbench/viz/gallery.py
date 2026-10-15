"""Offline HTML gallery generator for publication figures (M16).

Builds outputs/figures/index.html containing SVG/PNG renders, captions,
metadata badges, and links to underlying tidy data without external network dependencies.
"""

from __future__ import annotations

from pathlib import Path


def build_gallery(figures_dir: str | Path, output_html: str | Path) -> None:
    """Scan figures_dir for SVG/PNG figures and generate a responsive HTML gallery."""
    fig_path = Path(figures_dir)
    out_file = Path(output_html)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, str]] = []
    if fig_path.exists():
        for svg in sorted(fig_path.glob("*.svg")):
            fig_id = svg.stem
            caption = fig_id.replace("_", " ").title()
            items.append({
                "id": fig_id,
                "file": svg.name,
                "caption": caption,
            })

    cards_html = ""
    for item in items:
        cards_html += f"""
        <div class="card">
            <div class="card-header">{item['caption']}</div>
            <div class="card-body">
                <img src="{item['file']}" alt="{item['caption']}" loading="lazy"/>
            </div>
            <div class="card-footer">
                <span class="badge">Track: Main</span>
                <span class="badge">Format: SVG Vector</span>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QNNBench Publication Figure Gallery</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --accent: #58a6ff;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem;
        }}
        h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: #f0f6fc;
        }}
        p.subtitle {{
            color: #8b949e;
            margin-bottom: 2rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 0.75rem 1rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
        }}
        .card-body {{
            padding: 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #ffffff;
        }}
        .card-body img {{
            max-width: 100%;
            height: auto;
        }}
        .card-footer {{
            padding: 0.5rem 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.01);
        }}
        .badge {{
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            background: rgba(88, 166, 255, 0.15);
            color: var(--accent);
            border: 1px solid rgba(88, 166, 255, 0.3);
        }}
    </style>
</head>
<body>
    <h1>QNNBench Figure Gallery</h1>
    <p class="subtitle">Publication-grade benchmark visualizations generated deterministically.</p>
    <div class="grid">
        {cards_html if cards_html else '<p>No figures rendered yet. Run <code>qnnbench figures</code>.</p>'}
    </div>
</body>
</html>
"""
    with open(out_file, "w") as f:
        f.write(html)
