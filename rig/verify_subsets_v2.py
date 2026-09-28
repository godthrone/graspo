#!/usr/bin/env python3
"""只读复核 subsets-v2/：文件集合、行数、sha256、来源、与 v2 manifest 的 subset_size 对齐。"""
import hashlib, json, pathlib, sys

S = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/<user>/<RUN_ROOT_NAME>/subsets-v2")
V2 = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/home/<user>/<RUN_ROOT_NAME>/tests/e2e/matrix54_manifest_v2.json")
man = json.loads((S / "subsets-v2.manifest.json").read_text())
recs = {r["tier_id"]: r for r in man["tiers"]}
files = sorted(p.name for p in S.iterdir() if p.is_file())
print(f"目录: {S}")
print(f"文件数: {len(files)}（期望 54 jsonl + 1 manifest = 55）")
print(f"jsonl: {sum(1 for f in files if f.endswith('.jsonl'))}  其他: {[f for f in files if not f.endswith('.jsonl')]}")
problems = []
by_size = {}
for tid, r in sorted(recs.items()):
    p = S / f"{tid}.jsonl"
    if not p.is_file():
        problems.append(f"{tid}: 缺文件"); continue
    raw = p.read_bytes()
    n = raw.count(b"\n"); h = hashlib.sha256(raw).hexdigest()
    by_size.setdefault((r["source"], n), []).append(tid)
    if n != r["subset_size"]:
        problems.append(f"{tid}: 行数 {n} != 清单 subset_size {r['subset_size']}")
    if h != r["sha256"]:
        problems.append(f"{tid}: sha256 不符")
extra = [f for f in files if f.endswith(".jsonl") and f[:-6] not in recs]
if extra:
    problems.append(f"多余文件: {extra}")
print("\n逐档分组（source, 行数）→ 档数:")
for (src, n), tids in sorted(by_size.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
    print(f"  {str(src):8s} {n:4d} 行 : {len(tids):2d} 档  {','.join(tids) if len(tids)<=6 else tids[0]+'..'+tids[-1]}")
print(f"\n来源核对: formal={sum(1 for r in recs.values() if r['source']=='formal')} "
      f"mini={sum(1 for r in recs.values() if r['source']=='mini')} "
      f"not_applicable={len(man['not_applicable_tiers'])}")
print("源文件指纹:", json.dumps(man["source_files"], ensure_ascii=False))
# 与 v2 manifest 交叉核对（若在 228 上可读）
if V2.is_file():
    v2 = json.loads(V2.read_text())
    v2recs = {t["tier_id"]: t for t in v2["tiers"]}
    for tid, r in recs.items():
        t = v2recs.get(tid)
        if not t:
            problems.append(f"{tid}: v2 manifest 里没有"); continue
        if int(t["data"]["subset_size"]) != int(r["subset_size"]):
            problems.append(f"{tid}: v2 manifest subset_size {t['data']['subset_size']} != 子集清单 {r['subset_size']}")
        want_src = t["data"].get("source")
        if want_src is not None and want_src != r["source"]:
            problems.append(f"{tid}: v2 manifest source {want_src} != 子集清单 {r['source']}")
    print(f"\n与 v2 manifest 交叉核对: {V2}（54 档 subset_size/source）")
else:
    print(f"\n与 v2 manifest 交叉核对: **跳过**（{V2} 不存在）")
print("\n结论:", "OK（无失配）" if not problems else f"FAILED（{len(problems)} 项）")
for p in problems:
    print("  -", p)
sys.exit(0 if not problems else 1)
