import os
import re
import json
import tempfile
import torch
import fitz  # PyMuPDF
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


# ── Markdown table parsing → structured JSON ──
#
# The model represents tables using standard markdown table syntax:
#   | Col A | Col B |
#   | --- | --- |
#   | val1 | val2 |
# This is parsed into a list of dicts (one per row) plus any leftover
# free text, so a separate script can turn it into an Excel workbook
# without needing to re-parse markdown itself.

TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def parse_markdown_tables(md_text):
    lines = md_text.splitlines()
    tables = []
    i = 0
    table_idx = 0

    while i < len(lines):
        line = lines[i]
        if TABLE_ROW_RE.match(line):
            if i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
                header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
                i += 2
                rows = []
                while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                    row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    row_cells = (row_cells + [""] * len(header_cells))[: len(header_cells)]
                    rows.append(dict(zip(header_cells, row_cells)))
                    i += 1
                table_idx += 1
                tables.append({
                    "table_index": table_idx,
                    "headers": header_cells,
                    "rows": rows,
                })
                continue
        i += 1

    return tables


def parse_narrative_text(md_text):
    """Strip out table blocks, keep everything else (contract clauses, terms, etc.)."""
    lines = md_text.splitlines()
    narrative_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if TABLE_ROW_RE.match(line) and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            i += 2
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                i += 1
            continue
        narrative_lines.append(line)
        i += 1
    return "\n".join(narrative_lines).strip()


def build_structured_output(source_path, md_text):
    return {
        "source_file": os.path.basename(source_path),
        "tables": parse_markdown_tables(md_text),
        "narrative_text": parse_narrative_text(md_text),
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