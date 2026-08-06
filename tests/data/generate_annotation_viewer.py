#!/usr/bin/env python3
"""Generate an HTML visualization of annotation_testset.jsonl.

Each character in the completion is color-coded by its annotation label:
  S (Structure)  = green   — 结构标记，训练
  V (Value)      = blue    — 值，训练
  W (Waste)      = gray    — 废话/前导/尾随，不训练
  T (Think)      = purple  — 思考内容，不训练
  E (Error)      = red     — 首个错误字符
  D (Discard)    = pink    — 错误之后丢弃，不训练
"""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / "annotation_testset.jsonl"
OUTPUT_FILE = Path(__file__).parent / "annotation_viewer.html"

LABEL_COLORS = {
    "S": "#2e7d32",  # green — structure
    "V": "#1565c0",  # blue — value
    "W": "#9e9e9e",  # gray — waste
    "T": "#7b1fa2",  # purple — think
    "E": "#c62828",  # red — error
    "D": "#b7a520",  # dark yellow — discard (after error)
}

LABEL_NAMES = {
    "S": "Structure",
    "V": "Value",
    "W": "Waste",
    "T": "Think",
    "E": "Error",
    "D": "Discard",
}

CSS = """
:root {
    --bg: #f5f6fa;
    --card-bg: #ffffff;
    --text: #1a1a2e;
    --muted: #6b7280;
    --border: #d1d5db;
    --accent: #e5e7eb;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 20px;
    line-height: 1.6;
}
h1 { text-align: center; margin-bottom: 8px; font-size: 1.6em; }
.subtitle { text-align: center; color: var(--muted); margin-bottom: 24px; font-size: 0.9em; }

/* Summary bar */
.summary {
    display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;
    margin-bottom: 28px;
}
.summary .stat {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 18px; text-align: center; min-width: 80px;
}
.summary .stat .count { font-size: 1.5em; font-weight: bold; }
.summary .stat .label { font-size: 0.78em; color: var(--muted); }

/* Filters */
.filters {
    display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;
    margin-bottom: 20px; align-items: center;
}
.filters button, .filters select {
    padding: 6px 16px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--card-bg); color: var(--text); cursor: pointer;
    font-size: 0.88em; transition: all 0.15s;
}
.filters button:hover, .filters select:hover { border-color: #9ca3af; }
.filters button.active { background: #2563eb; border-color: #2563eb; color: #fff; }
.filters button.reset { background: #fef3c7; border-color: #d97706; color: #92400e; }

/* Search */
.search-box {
    display: flex; justify-content: center; margin-bottom: 20px;
}
.search-box input {
    width: 100%; max-width: 500px; padding: 8px 14px;
    border-radius: 8px; border: 1px solid var(--border);
    background: var(--card-bg); color: var(--text); font-size: 0.9em;
    outline: none;
}
.search-box input:focus { border-color: #2563eb; }

/* Card */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 16px;
    overflow: hidden;
    transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.12); }
.card-header {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 12px 16px; cursor: pointer; user-select: none;
    background: rgba(0,0,0,0.02);
}
.card-header:hover { background: rgba(0,0,0,0.04); }
.card-id {
    font-weight: bold; font-size: 1.05em; min-width: 36px;
    color: #1d4ed8;
}
.card-case {
    font-size: 0.92em; flex: 1; min-width: 150px;
}
.card-type {
    font-size: 0.75em; padding: 2px 10px; border-radius: 12px;
    background: var(--accent); color: #374151;
}
.card-correct {
    font-size: 0.75em; padding: 2px 10px; border-radius: 12px;
    font-weight: bold;
}
.card-correct.yes { background: #dcfce7; color: #166534; }
.card-correct.no { background: #fee2e2; color: #991b1b; }
.card-error {
    font-size: 0.78em; color: #b91c1c;
}
.card-notes {
    font-size: 0.8em; color: var(--muted); padding: 0 16px 10px;
}
.card-body { display: none; padding: 0 16px 16px; }
.card.open .card-body { display: block; }
.card-toggle { font-size: 0.8em; color: var(--muted); margin-left: auto; }

/* Completion display */
.completion-block {
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 0.88em; line-height: 1.75;
    white-space: pre-wrap; word-break: break-all;
    background: #fafafa; border-radius: 8px; padding: 14px 16px;
    border: 1px solid var(--border);
    overflow-x: auto;
    max-height: 400px; overflow-y: auto;
}
.completion-block span.char {
    border-radius: 2px;
    padding: 0 0.5px;
}
.completion-block span.char.S { background: rgba(46,125,50,0.35); }
.completion-block span.char.V { background: rgba(21,101,192,0.35); }
.completion-block span.char.W { background: rgba(158,158,158,0.25); }
.completion-block span.char.T { background: rgba(123,31,162,0.3); }
.completion-block span.char.E { background: rgba(220,38,38,0.65); font-weight: bold; }
.completion-block span.char.D { background: rgba(183,165,32,0.4); }
.completion-block span.char.space-char { }

/* Ground truth */
.gt-block {
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 0.82em; background: rgba(0,0,0,0.02);
    border: 1px dashed var(--border); border-radius: 6px;
    padding: 8px 12px; margin-top: 8px; color: var(--muted);
    word-break: break-all;
}
.gt-label { font-size: 0.72em; color: #6a6a8a; margin-bottom: 2px; }

/* Legend */
.legend {
    display: flex; flex-wrap: wrap; gap: 14px; justify-content: center;
    margin-bottom: 22px; font-size: 0.82em;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.legend-swatch {
    width: 14px; height: 14px; border-radius: 3px; display: inline-block;
}
.legend-label { color: var(--muted); }

/* Empty completion */
.empty-note { color: var(--muted); font-style: italic; padding: 8px; }

/* Position ruler */
.ruler {
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 0.7em; color: #9ca3af; padding: 2px 16px;
    white-space: pre-wrap; word-break: break-all;
    user-select: none;
}
</style>
"""

JS = """
// Toggle card open/close
document.querySelectorAll('.card-header').forEach(h => {
    h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
});

// Filter by type
let activeType = 'all';
let searchText = '';

function applyFilters() {
    document.querySelectorAll('.card').forEach(card => {
        const type = card.dataset.type;
        const text = card.dataset.searchText;
        const typeMatch = activeType === 'all' || type === activeType;
        const searchMatch = !searchText || text.includes(searchText);
        card.style.display = (typeMatch && searchMatch) ? '' : 'none';
    });
}

document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeType = btn.dataset.type;
        applyFilters();
    });
});

document.getElementById('searchInput').addEventListener('input', (e) => {
    searchText = e.target.value.toLowerCase();
    applyFilters();
});

document.getElementById('resetBtn').addEventListener('click', () => {
    activeType = 'all';
    searchText = '';
    document.getElementById('searchInput').value = '';
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.filter-btn[data-type="all"]').classList.add('active');
    applyFilters();
});
"""


def load_data():
    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_char_spans(completion, annotation):
    """Build HTML spans for each character colored by annotation label."""
    if not completion:
        return '<span class="empty-note">(empty completion)</span>'

    spans = []
    for i, (char, label) in enumerate(zip(completion, annotation)):
        # Escape HTML entities
        if char == "<":
            char_esc = "&lt;"
        elif char == ">":
            char_esc = "&gt;"
        elif char == "&":
            char_esc = "&amp;"
        elif char == "\n":
            char_esc = "↵<br>"
        elif char == " ":
            char_esc = "&nbsp;"
        elif char == "\t":
            char_esc = "&nbsp;&nbsp;&nbsp;&nbsp;"
        else:
            char_esc = char

        # Use a special class for spaces to keep them visible but not colorful
        space_class = "space-char" if char in (" ", "\t") else ""
        spans.append(f'<span class="char {label} {space_class}" title="pos={i} label={LABEL_NAMES.get(label, label)}">{char_esc}</span>')

    # Handle trailing characters in completion beyond annotation length
    if len(completion) > len(annotation):
        for i in range(len(annotation), len(completion)):
            char = completion[i]
            if char == "<":
                char_esc = "&lt;"
            elif char == ">":
                char_esc = "&gt;"
            elif char == "&":
                char_esc = "&amp;"
            elif char == "\n":
                char_esc = "↵<br>"
            elif char == " ":
                char_esc = "&nbsp;"
            elif char == "\t":
                char_esc = "&nbsp;&nbsp;&nbsp;&nbsp;"
            else:
                char_esc = char
            spans.append(f'<span class="char ?" style="background:rgba(255,255,0,0.4)" title="pos={i} label=UNANNOTATED">{char_esc}</span>')

    return "".join(spans)


def build_ruler(completion):
    """Build a position ruler (every 10 chars)."""
    if not completion:
        return ""
    lines = completion.split("\n")
    ruler_lines = []
    pos = 0
    for line in lines:
        ruler_chars = []
        for j in range(len(line)):
            if pos % 10 == 0 and pos > 0:
                ruler_chars.append(str(pos // 10 % 10))
            elif pos % 10 == 5:
                ruler_chars.append("·")
            else:
                ruler_chars.append(" ")
            pos += 1
        # newline
        pos += 1
        ruler_lines.append("".join(ruler_chars))
    return "\n".join(ruler_lines)


def build_html(records):
    # Stats
    total = len(records)
    type_counts = {}
    case_correct = {"yes": 0, "no": 0}
    all_labels = []
    for r in records:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
        case_correct[r.get("correct", "?")] = case_correct.get(r.get("correct", "?"), 0) + 1
        all_labels.extend(list(r.get("annotation", "")))

    label_counts = {l: all_labels.count(l) for l in LABEL_COLORS}

    cards_html = []
    for r in records:
        completion = r.get("completion", "")
        annotation = r.get("annotation", "")
        char_spans = build_char_spans(completion, annotation)
        ruler = build_ruler(completion)
        error_pos = r.get("error_pos", "")
        correct_class = "yes" if r.get("correct") == "yes" else "no"
        search_text = (r["id"] + " " + r["case"] + " " + r.get("notes", "") + " " + completion).lower()

        # Build error position display
        error_html = ""
        if error_pos != "":
            error_html = f'<span class="card-error">⚠ Error at pos {error_pos}</span>'

        cards_html.append(f"""
<div class="card" data-type="{r['type']}" data-search-text="{search_text}">
    <div class="card-header">
        <span class="card-id">{r['id']}</span>
        <span class="card-case">{r['case']}</span>
        <span class="card-type">{r['type']}</span>
        <span class="card-correct {correct_class}">{r.get('correct', '?')}</span>
        {error_html}
        <span class="card-toggle">▼</span>
    </div>
    <div class="card-body">
        <div class="card-notes">📝 {r.get('notes', '')}</div>
        <div class="ruler">{ruler}</div>
        <div class="completion-block">{char_spans}</div>
        <div class="gt-label">Ground Truth:</div>
        <div class="gt-block">{r.get('ground_truth', '')}</div>
    </div>
</div>""")

    # Legend
    legend_items = []
    for label, color in LABEL_COLORS.items():
        name = LABEL_NAMES[label]
        count = label_counts.get(label, 0)
        legend_items.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>'
            f'<span style="color:{color};font-weight:bold">{label}</span>'
            f'<span class="legend-label">= {name} ({count})</span></div>'
        )

    type_filter_buttons = ['<button class="filter-btn active" data-type="all">All ({})</button>'.format(total)]
    for t, c in sorted(type_counts.items()):
        type_filter_buttons.append(f'<button class="filter-btn" data-type="{t}">{t} ({c})</button>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Annotation Testset Viewer</title>
<style>{CSS}</style>
</head>
<body>
<h1>🎨 Annotation Testset Viewer</h1>
<p class="subtitle">{total} records &middot; {type_counts.get('tool_call', 0)} tool_call &middot; {type_counts.get('json', 0)} json</p>

<div class="summary">
    <div class="stat"><div class="count">{total}</div><div class="label">Total</div></div>
    <div class="stat"><div class="count">{case_correct.get('yes', 0)}</div><div class="label">Correct</div></div>
    <div class="stat"><div class="count">{case_correct.get('no', 0)}</div><div class="label">Incorrect</div></div>
    <div class="stat"><div class="count">{label_counts.get('E', 0)}</div><div class="label">Errors (E)</div></div>
</div>

<div class="legend">{"".join(legend_items)}</div>

<div class="filters">
    {"".join(type_filter_buttons)}
    <button class="reset" id="resetBtn">Reset</button>
</div>

<div class="search-box">
    <input type="text" id="searchInput" placeholder="Search by ID, case name, notes, or content...">
</div>

<div id="cards">
{"".join(cards_html)}
</div>

<script>
const LABEL_NAMES = {json.dumps(LABEL_NAMES)};
{JS}
</script>
</body>
</html>"""


def main():
    records = load_data()
    html = build_html(records)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Generated {OUTPUT_FILE} with {len(records)} records")


if __name__ == "__main__":
    main()