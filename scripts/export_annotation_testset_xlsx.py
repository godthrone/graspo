"""导出标注测试数据集为 xlsx（供 Excel 人工查看）。

用法：uv run --frozen python scripts/export_annotation_testset_xlsx.py

输出：tests/data/annotation_testset.xlsx
- 每行一条用例：id / type / case / completion / ground_truth / annotation / correct / error_pos / notes
- completion 与 annotation 保留换行（Excel 单元格内换行，需开启自动换行查看）
"""

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment

TESTSET = Path(__file__).resolve().parents[1] / "tests" / "data" / "annotation_testset.jsonl"
OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "annotation_testset.xlsx"

FIELDS = ["id", "type", "case", "completion", "ground_truth", "annotation", "correct", "error_pos", "notes"]


def main() -> None:
    with open(TESTSET, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    wb = Workbook()
    ws = wb.active
    ws.title = "annotation_testset"
    ws.append(FIELDS)

    wrap = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        ws.append([row.get(field, "") for field in FIELDS])

    for col in range(1, len(FIELDS) + 1):
        ws.column_dimensions[chr(ord("A") + col - 1)].width = 22
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = wrap

    wb.save(OUT)
    print(f"written {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
