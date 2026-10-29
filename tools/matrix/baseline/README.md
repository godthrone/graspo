# 模板基线（matrix-template-baseline.html）

## 这是什么
**当前模板 + 空白种子数据件**渲染出的基线人读件。它是**写范围闸门（§22）的比对基线**。

## 为什么有它（2026-10-28 主席批准解冻模板）
宪法 §17.7「边界声明义务」要求人读件正面陈述三点（已跑通 / 达标是可用证据 / 使用者效果不承诺），
而冻结原件只做了「效果不保证」的反面表述 ⇒ 主席批准**解冻模板**，在第 1 章之后补入「边界声明」小节
（文案由主席定稿，逐字照抄）。

冻结原件 `docs/GRASPO训练框架功能测试报告.html` 因此退为**历史基准件**：
- 它的三锁（sha256 `b8ed080d…` / 77079 B / 444）**仍然有效、仍被 `verify_dated.sh` ① 与 `new_round.sh` ① 逐项断言**；
- 但它**不再**充当写范围闸门的比对基线（否则每次回填都会因为「边界声明只存在于新模板里」而被判越界）。

## 生成方式（唯一通道：渲染器，⛔ 不手改）
```bash
.venv/bin/python tools/matrix/render_matrix.py \
  --template tools/matrix/template.html \
  --data docs/capability-matrix.data.json \
  --out tools/matrix/baseline/matrix-template-baseline.html
```

## 记录在案的指纹（改模板/改种子数据后必须同步重算，见 `tools/matrix/paths.py`）
| 项 | sha256 |
|---|---|
| 本基线 html | `TEMPLATE_BASELINE_HTML_SHA256`（paths.py 唯一真相源；闸门每次比对前核一次，不符即 fail-closed） |
| 模板 `tools/matrix/template.html` | 见提交说明（重算本基线时一并记录） |
| 种子数据件 `docs/capability-matrix.data.json` | 同上 |

## 与冻结原件的差异（机器证明）
见同目录 `diff-vs-frozen-original.txt`：**1 个 hunk（+145,18）、新增 12 行、删除 0 行**，内容只有「边界声明」小节
+ 两行来源注释，别无其他差异。

## 机核断言（`verify_dated.sh` ②b，⛔ 硬失败、不降级为警告）
`tools/matrix/check_baseline_declaration.py` 直接比对**冻结原件 ↔ 本基线**，判据：
1. 必须**恰好 1 个 hunk**、**只有新增、删除 0 行**（用「最长公共前缀/后缀」精确判定，不靠 diff 文本）；
2. 新增行数 == 模板里「边界声明」块的行数（现为 **10 行**），且**逐行逐字符相同**——期望块从
   `tools/matrix/template.html` **现抽**，⛔ 不在校验器里复制文案；
3. 插入位置必须紧跟第 1 章导语列表的 `</ul>`，且到该章 `</section>` 之间不得出现矩阵表。
失败时打印**逐行差异**（unified diff）并 `rc≠0`；`verify_dated.sh` 记为 `✗ FAIL`（硬失败）。
⭐ 不依赖 `diff-vs-frozen-original.txt`：那份文件只是给人看的静态证据，删掉它不影响断言成立。

## 纪律
- ⛔ 不得手改本文件；只能由上面的渲染命令重生成。
- ⛔ 不得改本文件的差异面：`verify_dated.sh` ②b 要求「冻结原件 ↔ 本文件」之差**恰为**模板里那块
  边界声明（1 hunk / 删除 0 行）。多一句、少一句、挪个位置，②b 都会红并打印逐行差异。
- ⛔ 不得为了让某次回填通过而改本文件 —— 闸门会先核 sha256，改了立即 fail-closed。
- 改模板后：重渲染本基线 → 更新 `paths.py` 的 `TEMPLATE_BASELINE_HTML_SHA256` → 重渲染当轮副本 → 跑
  `verify_matrix.py` 与 `verify_dated.sh`。
