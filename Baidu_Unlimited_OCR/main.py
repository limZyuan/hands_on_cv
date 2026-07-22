"""
Usage:
    Usage: python main.py <image_or_pdf_path> [gundam|base]
"""

import os
import re
import json
import html
import tempfile
import torch
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoTokenizer

# Point this at your LOCAL folder, not the HF repo id
MODEL_PATH = "model"

# Belt-and-suspenders: force transformers to never reach out to the network
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()

def ocr_single_image(image_path, output_dir="./outputs", mode="gundam"):
    """mode: 'gundam' (base_size=1024, image_size=640, crop_mode=True)
             'base'   (base_size=1024, image_size=1024, crop_mode=False)"""
    if mode == "gundam":
        base_size, image_size, crop_mode = 1024, 640, True
    else:
        base_size, image_size, crop_mode = 1024, 1024, False

    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=image_path,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=True,
    )

def pdf_to_images(pdf_path, dpi=300):
    """Render each PDF page to a PNG in a temp dir. Returns list of image paths."""
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def ocr_pdf(pdf_path, output_dir="./outputs", dpi=300):
    """
    Convert a PDF contract to images and run multi-page parsing.
    Writes result.md to output_dir (model side effect), then parses it
    into structured.json for downstream Excel conversion.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = pdf_to_images(pdf_path, dpi=dpi)

    model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=image_paths,
        output_path=output_dir,
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=True,
    )

    result_path = os.path.join(output_dir, "result.md")
    with open(result_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    structured = build_structured_output(pdf_path, md_text)
    structured_path = os.path.join(output_dir, "structured.json")
    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"Raw markdown:    {result_path}")
    print(f"Structured JSON: {structured_path}")
    return structured


# ── Table parsing → structured JSON ──
#
# The model can emit tables in two formats:
#   1. Standard markdown pipe tables:  | Col A | Col B |
#                                      | --- | --- |
#                                      | val1 | val2 |
#   2. Raw embedded HTML tables:       <table><tr><td colspan="2">...</td></tr></table>
#      (common for contract/form-style headers — party details, key terms, etc.)
#
# Both are parsed into a common "grid" representation: a list of rows,
# each row a list of [text, colspan] pairs. Colspan is preserved (not
# flattened/duplicated) so a downstream Excel script can create real
# merged cells. This grid schema is what json_to_excel.py expects.

TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def clean_text(text):
    """Unescape HTML entities and strip common OCR/LaTeX artifacts."""
    text = html.unescape(text)
    # \( ^{th} \)  ->  th      (superscript ordinal markers picked up as LaTeX)
    text = re.sub(r"\\\(\s*\^\{(\w+)\}\s*\\\)", r"\1", text)
    # Any remaining stray \( ... \) LaTeX inline-math wrappers -> keep inner content
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"\1", text)
    return text.strip()


def parse_html_table(table_html):
    """Parse one <table>...</table> block into a grid of [text, colspan] rows."""
    soup = BeautifulSoup(table_html, "html.parser")
    grid = []
    for tr in soup.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            colspan = int(cell.get("colspan", 1) or 1)
            text = clean_text(cell.get_text(separator=" ", strip=True))
            row.append([text, colspan])
        if row:
            grid.append(row)
    return grid


def parse_markdown_pipe_tables(md_text):
    """Returns a list of grids for markdown pipe tables (colspan always 1)."""
    lines = md_text.splitlines()
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if TABLE_ROW_RE.match(line) and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            header_cells = [clean_text(c.strip()) for c in line.strip().strip("|").split("|")]
            grid = [[[c, 1] for c in header_cells]]
            i += 2
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                row_cells = [clean_text(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                grid.append([[c, 1] for c in row_cells])
                i += 1
            results.append(grid)
            continue
        i += 1
    return results


def grid_is_uniform_data_table(grid):
    """Heuristic: a 'real' data table has a consistent column count and no colspans."""
    col_counts = {sum(c[1] for c in row) for row in grid}
    has_colspan = any(c[1] > 1 for row in grid for c in row)
    return len(col_counts) == 1 and not has_colspan and len(grid) > 1


def extract_tables_and_narrative(md_text):
    """
    Extracts both HTML tables and markdown pipe tables (in that priority),
    and returns (tables, narrative_text) where tables is a list of dicts:
        {"table_index": int, "grid": [[[text, colspan], ...], ...], "is_data_table": bool}
    """
    tables = []

    html_grids = [parse_html_table(m.group(0)) for m in HTML_TABLE_RE.finditer(md_text)]
    text_without_html_tables = HTML_TABLE_RE.sub("\n", md_text)
    md_pipe_grids = parse_markdown_pipe_tables(text_without_html_tables)

    for grid in html_grids + md_pipe_grids:
        if grid:
            tables.append(grid)

    # Narrative: strip HTML tables, strip markdown pipe table lines, clean remaining text
    narrative = HTML_TABLE_RE.sub("", md_text)
    narrative_lines = narrative.splitlines()
    kept_lines = []
    j = 0
    while j < len(narrative_lines):
        line = narrative_lines[j]
        if TABLE_ROW_RE.match(line) and j + 1 < len(narrative_lines) and TABLE_SEP_RE.match(narrative_lines[j + 1]):
            j += 2
            while j < len(narrative_lines) and TABLE_ROW_RE.match(narrative_lines[j]):
                j += 1
            continue
        kept_lines.append(line)
        j += 1
    narrative_text = clean_text("\n".join(kept_lines))
    narrative_text = re.sub(r"\n{3,}", "\n\n", narrative_text).strip()

    tables_out = [
        {"table_index": idx, "grid": grid, "is_data_table": grid_is_uniform_data_table(grid)}
        for idx, grid in enumerate(tables, start=1)
    ]
    return tables_out, narrative_text


def build_structured_output(source_path, md_text):
    tables, narrative_text = extract_tables_and_narrative(md_text)
    return {
        "source_file": os.path.basename(source_path),
        "tables": tables,
        "narrative_text": narrative_text,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_or_pdf_path> [gundam|base]")
        sys.exit(1)
    input_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "gundam"

    if input_path.lower().endswith(".pdf"):
        ocr_pdf(input_path, output_dir="./outputs")
    else:
        ocr_single_image(input_path, output_dir="./outputs", mode=mode)