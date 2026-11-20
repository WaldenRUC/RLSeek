"""Convert official RAGTruth JSONL for an unofficial RLSeek compatibility check."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


TASKS = ("QA", "Data2txt", "Summary")
SOURCES = {"train": "ragtruth", "test": "ragtruth-test"}
SCHEMA = pa.schema([
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("data_source", pa.string()),
    ("reward_model", pa.struct([("ground_truth", pa.string())])),
    ("extra_info", pa.struct([
        ("index", pa.int64()),
        ("response", pa.string()),
        ("reference", pa.string()),
        ("binary_label", pa.int64()),
        ("response_id", pa.string()),
        ("source_id", pa.string()),
        ("task_type", pa.string()),
        ("quality", pa.string()),
    ])),
], metadata={b"provenance": b"Unofficial RLSeek compatibility conversion from RAGTruth"})


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def reference_text(source):
    info = source["source_info"]
    if source["task_type"] == "QA":
        return info["passages"]
    if source["task_type"] == "Data2txt":
        return json.dumps(info, ensure_ascii=False, indent=2)
    return info


def make_row(record, source, index):
    reference = reference_text(source)
    question = ""
    if source["task_type"] == "QA":
        question = f"\n\nQuestion:\n{source['source_info']['question']}"
    prompt = (
        "Identify spans in the candidate response that are contradicted by or unsupported "
        "by the reference. For each reasoning step, use a heading starting with ## Step "
        "and cite supporting reference text verbatim in <quote>...</quote>. "
        'Finish with a JSON object with the key "hallucination list" and a list of exact '
        "substrings from the candidate response. Use an empty list if no span qualifies."
        f"{question}\n\nReference:\n{reference}\n\nCandidate response:\n{record['response']}"
    )
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "data_source": SOURCES[record["split"]],
        "reward_model": {"ground_truth": repr(record["labels"])},
        "extra_info": {
            "index": index,
            "response": record["response"],
            "reference": reference,
            "binary_label": int(bool(record["labels"])),
            "response_id": record["id"],
            "source_id": record["source_id"],
            "task_type": source["task_type"],
            "quality": record["quality"],
        },
    }


def convert(raw, output, per_group):
    sources = {}
    for source in read_jsonl(raw / "source_info.jsonl"):
        source_id = source["source_id"]
        if source_id in sources or source["task_type"] not in TASKS:
            raise ValueError(f"Duplicate or unsupported source: {source_id}")
        sources[source_id] = source

    source_splits = {}
    response_ids = set()
    totals = Counter()
    picks = Counter()
    rows = {split: [] for split in SOURCES}
    spans = 0
    for record in read_jsonl(raw / "response.jsonl"):
        rid, sid, split = record["id"], record["source_id"], record["split"]
        if rid in response_ids or sid not in sources or split not in SOURCES:
            raise ValueError(f"Duplicate response, missing source or unsupported split: {rid}")
        if source_splits.setdefault(sid, split) != split:
            raise ValueError(f"Source occurs in both splits: {sid}")
        response_ids.add(rid)
        totals[split] += 1
        source = sources[sid]
        for label in record["labels"]:
            start, end = label["start"], label["end"]
            if not (0 <= start < end <= len(record["response"])):
                raise ValueError(f"Invalid span bounds in response {rid}: {start}, {end}")
            if record["response"][start:end] != label["text"]:
                raise ValueError(f"Span text mismatch in response {rid}: {start}, {end}")
            spans += 1
        group = (split, source["task_type"], bool(record["labels"]))
        if per_group is None or picks[group] < per_group:
            rows[split].append(make_row(record, source, len(rows[split])))
            picks[group] += 1

    for split in SOURCES:
        for task in TASKS:
            for has_labels in (False, True):
                if picks[split, task, has_labels] < (per_group or 1):
                    raise ValueError(f"Insufficient rows for {split}/{task}/{has_labels}")

    output.mkdir(parents=True, exist_ok=False)
    for split, records in rows.items():
        pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), output / f"{split}.parquet")
        print(f"{split}: input={totals[split]}, written={len(records)}")
    print(f"Audited {len(source_splits)} joined sources and {spans} spans.")
    print("Written by (split, task, has_labels):", sorted(picks.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--per-group", type=int, help="First N rows per split/task/label presence; omit for all")
    args = parser.parse_args()
    if args.per_group is not None and args.per_group < 1:
        parser.error("--per-group must be positive")
    convert(args.raw, args.output, args.per_group)


if __name__ == "__main__":
    main()
