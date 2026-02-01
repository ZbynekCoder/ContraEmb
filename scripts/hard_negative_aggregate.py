import csv
from collections import OrderedDict
import numpy as np


INPUT_CSV = "data/arguana_validation_final.csv"
OUTPUT_CSV = "data/arguana_validation_aggregate.csv"

QUERY_COL = 0
GOLD_COL = 1
HARD_COL = 2


def aggregate_csv(input_csv, output_csv):
    """
    Aggregate (query, gold, hard) rows into one row per (query, gold),
    expanding hard negatives into multiple columns: hard_0, hard_1, ...
    """

    # (query, gold) -> list of hard negatives
    data = OrderedDict()

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if len(row) <= max(QUERY_COL, GOLD_COL, HARD_COL):
                raise ValueError(
                    f"Row {row_idx} has too few columns: {len(row)}"
                )

            query = row[QUERY_COL].strip()
            gold = row[GOLD_COL].strip()
            hard = row[HARD_COL].strip()

            key = (query, gold)
            if key not in data:
                data[key] = []

            if hard != "":
                data[key].append(hard)

    # sanity check
    if not data:
        raise RuntimeError("No data loaded from CSV.")

    max_hard = max(len(hards) for hards in data.values())

    print(f"Loaded {len(data)} unique (query, gold) pairs")

    hard_counts = [len(v) for v in data.values()]
    print("Hard negatives distribution:")
    print("  min:", min(hard_counts))
    print("  max:", max(hard_counts))
    print("  mean:", np.mean(hard_counts))

    header = ["query", "gold"] + [f"hard_{i}" for i in range(max_hard)]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        first_skip = False

        for (query, gold), hards in data.items():
            if not first_skip:
                first_skip = True
                continue

            row = [query, gold]
            row.extend(hards)
            row.extend([""] * (max_hard - len(hards)))
            writer.writerow(row)

    print(f"Aggregated CSV written to: {output_csv}")


if __name__ == "__main__":
    aggregate_csv(INPUT_CSV, OUTPUT_CSV)
