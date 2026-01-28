import json
from datetime import datetime

DIR = "case_study/case_study_low_overlap"
IN = f"{DIR}.jsonl"
OUT = f"{DIR}.md"

def fmt_row(x):
    flag = "✓" if x["is_gold"] else " "
    return f"| {x['rank']:>2}. [{flag}] {x['docid']} | {x['snippet']} |"

with open(IN, "r", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as w:
    for line in f:
        ex = json.loads(line)
        w.write(f"\n## Query: {ex['query']}\n\n")
        w.write(f"- Gold doc: `{ex['gold_docid']}`\n")
        w.write(f"- Rank (ours): **{ex['rank_T']}**, Rank (bge): **{ex['rank_baseline']}**\n")
        w.write(f"- Overlap@K: **{ex['overlap@K']:.2f}**\n\n")

        w.write("### Ours: ⟨T(E(q)), E(d)⟩\n")
        w.write("| Rank | Snippet |\n|---|---|\n")
        for x in ex["topK_T"]:
            w.write(fmt_row(x) + "\n")

        w.write("\n### Baseline: ⟨E_base(q), E_base(d)⟩\n")
        w.write("| Rank | Snippet |\n|---|---|\n")
        for x in ex["topK_baseline"]:
            w.write(fmt_row(x) + "\n")

print(f"Wrote {OUT}")
