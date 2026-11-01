"""Write the English TyDi QA GoldP splits used in the project as JSONL files.

train: English train examples minus the dev set
dev:   400 English train examples, chosen with seed 1234
test:  English official validation split (held out)
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def english(split):
    rows = []
    for ex in split:
        if ex["id"].startswith("english"):
            rows.append({
                "id": ex["id"],
                "title": ex["title"],
                "context": ex["context"],
                "question": ex["question"],
                "answers": ex["answers"]["text"],
            })
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} examples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/tydiqa")
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    data = load_dataset("google-research-datasets/tydiqa", "secondary_task")
    train = english(data["train"])
    test = english(data["validation"])

    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    dev_ids = set(order[: args.dev_size])
    dev = [train[i] for i in sorted(dev_ids)]
    train = [row for i, row in enumerate(train) if i not in dev_ids]

    ids = [set(r["id"] for r in rows) for rows in (train, dev, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "splits overlap"

    out = Path(args.output_dir)
    write(out / "train.jsonl", train)
    write(out / "dev.jsonl", dev)
    write(out / "test.jsonl", test)


if __name__ == "__main__":
    main()
