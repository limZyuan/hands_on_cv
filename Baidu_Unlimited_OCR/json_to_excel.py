"""
json_to_excel.py

Companion script to extract_contract_pdf.py.
Reads structured.json (produced by that script) and writes an .xlsx
workbook: one sheet per table found in the contract, plus a
"Narrative" sheet holding any non-tabular text (clauses, terms, etc.).

Usage:
    python json_to_excel.py ./outputs/structured.json output.xlsx
"""

import sys
import json
import argparse
import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


def sanitize_sheet_name(name, used_names):
    """Excel sheet names: max 31 chars, no []:*?/\\, must be unique."""
    for ch in '[]:*?/\\':
        name = name.replace(ch, '')
    name = name[:31] or "Table"
    base = name
    i = 2
    while name in used_names:
        suffix = f"_{i}"
        name = (base[: 31 - len(suffix)]) + suffix
        i += 1
    used_names.add(name)
    return name


def write_excel(structured, output_path):
    tables = structured.get("tables", [])
    narrative = structured.get("narrative_text", "")
    source_file = structured.get("source_file", "")

    used_names = set()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if tables:
            for t in tables:
                df = pd.DataFrame(t["rows"], columns=t["headers"])
                sheet_name = sanitize_sheet_name(f"Table_{t['table_index']}", used_names)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Still produce a workbook even if no tables were detected
            pd.DataFrame({"Note": ["No tables were detected in the source document."]}).to_excel(
                writer, sheet_name="Table_1", index=False
            )

        if narrative.strip():
            narrative_df = pd.DataFrame({"Contract Text": narrative.split("\n")})
            narrative_df.to_excel(writer, sheet_name="Narrative", index=False)

        # Simple metadata sheet noting where this came from
        pd.DataFrame({"Field": ["Source PDF"], "Value": [source_file]}).to_excel(
            writer, sheet_name="Metadata", index=False
        )

    # Light formatting pass: bold headers, sensible column widths
    from openpyxl import load_workbook
    wb = load_workbook(output_path)
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = Font(bold=True, name="Arial")
        for col_idx, column_cells in enumerate(ws.columns, start=1):
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in column_cells)
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 60)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name="Arial")
    wb.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert structured.json (from extract_contract_pdf.py) into an Excel workbook.")
    parser.add_argument("json_path", help="Path to structured.json")
    parser.add_argument("output_xlsx", help="Path to write the .xlsx file")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        structured = json.load(f)

    write_excel(structured, args.output_xlsx)
    print(f"Wrote {args.output_xlsx}")


if __name__ == "__main__":
    main()
