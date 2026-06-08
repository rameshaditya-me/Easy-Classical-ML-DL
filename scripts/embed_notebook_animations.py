#!/usr/bin/env python3
"""Extract matplotlib animation HTML from notebook outputs for MkDocs.

nbconvert clears outputs when building the site, so animations saved in .ipynb
never appear in the generated Markdown. This script reads the original notebook
outputs (still in git), writes assets/*.html, and patches the converted .md
file with iframe embeds after the matching code cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ANIMATION_MARKERS = ("function Animation(", "_anim_slider", "_anim_img")


def _html_text(data: dict) -> str:
    html = data.get("text/html", "")
    if isinstance(html, list):
        return "".join(html)
    return html or ""


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


def animation_cells(notebook_path: Path) -> list[tuple[int, str, int]]:
    """Return (cell_index, asset_name, iframe_height) for each animation cell."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    assets_dir = notebook_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    found: list[tuple[int, str, int]] = []

    code_cell_index = -1
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code_cell_index += 1
        for output in cell.get("outputs", []):
            if output.get("output_type") != "display_data":
                continue
            html = _html_text(output.get("data", {}))
            if not is_animation_html(html):
                continue

            asset_name = f"{notebook_path.stem}_anim_{index}.html"
            (assets_dir / asset_name).write_text(html, encoding="utf-8")
            height = 520 if "max-width:520px" in html else 480
            found.append((code_cell_index, asset_name, height))
            break

    return found


def insert_embeds(markdown_path: Path, embeds: list[tuple[int, str, int]]) -> bool:
    """Insert iframe blocks after code cells that match animation cell indices."""
    if not embeds or not markdown_path.exists():
        return False

    embed_by_cell = {
        cell: iframe_markdown(asset, height=height)
        for cell, asset, height in embeds
    }

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
        if code_cell_index in embed_by_cell:
            output.append("\n")
            output.append(embed_by_cell[code_cell_index])
            changed = True

    if changed:
        markdown_path.write_text("".join(output), encoding="utf-8")
    return changed


def process_notebook(notebook_path: Path) -> int:
    embeds = animation_cells(notebook_path)
    if not embeds:
        return 0

    markdown_path = notebook_path.with_suffix(".md")
    if insert_embeds(markdown_path, embeds):
        print(f"patched {markdown_path} with {len(embeds)} animation embed(s)")
    else:
        print(f"warning: extracted {len(embeds)} animation(s) but could not patch {markdown_path}")
    return len(embeds)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "src/notebooks")
    total = 0
    for notebook in sorted(root.rglob("*.ipynb")):
        if notebook.name.endswith("-checkpoint.ipynb"):
            continue
        total += process_notebook(notebook)
    print(f"done ({total} animation(s) total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
