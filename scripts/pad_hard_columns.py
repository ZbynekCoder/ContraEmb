import csv
from pathlib import Path

FILES = [
    "../data/arguana_training_aggregate.csv",
    "../data/arguana_validation_aggregate.csv",
]
OUT_DIR = "../data/padded"

def get_max_hard_cols(path: str) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    hard_cols = [c for c in header if c.startswith("hard_")]
    if not hard_cols:
        return 0
    # hard_0..hard_k
    return max(int(c.split("_", 1)[1]) for c in hard_cols) + 1

def pad_file(in_path: str, out_path: str, target_k: int):
    with open(in_path, newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)

        # existing hard columns
        existing = [c for c in header if c.startswith("hard_")]
        existing_set = set(existing)

        base_cols = [c for c in header if not c.startswith("hard_")]
        new_header = base_cols + [f"hard_{i}" for i in range(target_k)]

        # map from column name to index
        idx = {c: i for i, c in enumerate(header)}

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(new_header)

            for row in reader:
                # preserve base cols
                out_row = [row[idx[c]] if c in idx and idx[c] < len(row) else "" for c in base_cols]
                # hard cols
                for i in range(target_k):
                    col = f"hard_{i}"
                    if col in existing_set:
                        out_row.append(row[idx[col]] if idx[col] < len(row) else "")
                    else:
                        out_row.append("")
                writer.writerow(out_row)

def main():
    max_k = 0
    for fp in FILES:
        max_k = max(max_k, get_max_hard_cols(fp))
    print(f"[INFO] Global max hard columns = {max_k}")

    for fp in FILES:
        out_fp = str(Path(OUT_DIR) / Path(fp).name)
        pad_file(fp, out_fp, max_k)
        print(f"[INFO] Wrote padded: {out_fp}")

if __name__ == "__main__":
    main()
