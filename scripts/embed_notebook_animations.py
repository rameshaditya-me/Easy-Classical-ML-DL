#!/usr/bin/env python3
"""Extract notebook figure outputs (PNG plots and HTML animations) for MkDocs.

nbconvert clears outputs when building the site, so plots and animations saved in
.ipynb files never appear in the generated Markdown. This script reads outputs
from the notebooks in git, writes assets/*, and patches each converted .md file.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ANIMATION_MARKERS = ("function Animation(", "_anim_slider", "_anim_img")
OUTPUT_TYPES = ("display_data", "execute_result")


def _html_text(data: dict) -> str:
    html = data.get("text/html", "")
    if isinstance(html, list):
        return "".join(html)
    return html or ""


def _png_bytes(data: dict) -> bytes | None:
    raw = data.get("image/png")
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = "".join(raw)
    return base64.b64decode(raw)


def is_animation_html(html: str) -> bool:
    return bool(html) and any(marker in html for marker in ANIMATION_MARKERS)


def iframe_markdown(asset_name: str, height: int = 480) -> str:
    return (
        '<div class="notebook-animation" markdown="0">\n'
        f'<iframe src="assets/{asset_name}" title="Notebook animation" '
        f'width="100%" height="{height}" frameborder="0" scrolling="no" '
        'loading="lazy"></iframe>\n'
        "</div>\n"
    )


def image_markdown(asset_name: str) -> str:
    return f'![Notebook figure](assets/{asset_name})\n'


def cell_embeds(notebook_path: Path) -> dict[int, list[str]]:
    """Map code-cell index -> markdown snippets to insert after that cell."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    assets_dir = notebook_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    embeds: dict[int, list[str]] = {}

    code_cell_index = -1
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code_cell_index += 1
        snippets: list[str] = []

        for output_idx, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") not in OUTPUT_TYPES:
                continue
            data = output.get("data", {})

            png = _png_bytes(data)
            if png is not None:
                asset_name = f"{notebook_path.stem}_fig_{index}_{output_idx}.png"
                (assets_dir / asset_name).write_bytes(png)
                snippets.append(image_markdown(asset_name))

            html = _html_text(data)
            if is_animation_html(html):
                asset_name = f"{notebook_path.stem}_anim_{index}.html"
                (assets_dir / asset_name).write_text(html, encoding="utf-8")
                height = 520 if "max-width:520px" in html else 480
                snippets.append(iframe_markdown(asset_name, height=height))

        if snippets:
            embeds[code_cell_index] = snippets

    return embeds


def insert_embeds(markdown_path: Path, embeds: dict[int, list[str]]) -> bool:
    """Insert figure/animation blocks after matching code cells."""
    if not embeds or not markdown_path.exists():
        return False

    lines = markdown_path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    code_cell_index = -1
    in_fence = False
    changed = False

    for line in lines:
        output.append(line)
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue

        if not in_fence:
            in_fence = True
            code_cell_index += 1
            continue

        in_fence = False
        if code_cell_index in embeds:
            output.append("\n")
            for snippet in embeds[code_cell_index]:
                output.append(snippet)
            changed = True

    if changed:
        markdown_path.write_text("".join(output), encoding="utf-8")
    return changed


def process_notebook(notebook_path: Path) -> tuple[int, int]:
    embeds = cell_embeds(notebook_path)
    if not embeds:
        return 0, 0

    n_fig = sum(
        1
        for snippets in embeds.values()
        for snippet in snippets
        if snippet.startswith("![")
    )
    n_anim = sum(
        1
        for snippets in embeds.values()
        for snippet in snippets
        if snippet.startswith("<div")
    )

    markdown_path = notebook_path.with_suffix(".md")
    if insert_embeds(markdown_path, embeds):
        print(
            f"patched {markdown_path} with {n_fig} figure(s) and {n_anim} animation(s)"
        )
    else:
        print(
            f"warning: extracted outputs for {notebook_path} but could not patch {markdown_path}"
        )
    return n_fig, n_anim


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "src/notebooks")
    total_fig = 0
    total_anim = 0
    for notebook in sorted(root.rglob("*.ipynb")):
        if notebook.name.endswith("-checkpoint.ipynb"):
            continue
        n_fig, n_anim = process_notebook(notebook)
        total_fig += n_fig
        total_anim += n_anim
    print(f"done ({total_fig} figure(s), {total_anim} animation(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
