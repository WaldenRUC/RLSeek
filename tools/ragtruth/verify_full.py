"""Audit an unsampled RAGTruth conversion with the RLSeek loader and reward code."""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.reward_score import default_compute_score
from verl.utils.tokenizer import hf_tokenizer

from prepare_ragtruth import SCHEMA, SOURCES, make_row, read_jsonl


def verify_split(split, records, sources, config, tokenizer):
    path = Path(config.train_files if split == "train" else config.val_files)
    limit = config.train_max_samples if split == "train" else config.val_max_samples
    assert limit == -1
    parquet = pq.ParquetFile(path)
    assert parquet.schema_arrow.equals(SCHEMA, check_metadata=False), parquet.schema_arrow
    assert parquet.schema_arrow.metadata[b"provenance"] == SCHEMA.metadata[b"provenance"]
    assert parquet.metadata.num_rows == len(records)

    lengths = {}
    reasons = {}
    spans = 0
    quality = Counter()
    label_counts = Counter()
    task_counts = Counter()
    rows = (row for batch in parquet.iter_batches() for row in batch.to_pylist())
    for index, (row, original) in enumerate(zip(rows, records, strict=True)):
        info = row["extra_info"]
        rid = original["id"]
        assert row == make_row(original, sources[original["source_id"]], index), rid
        ground_truth = row["reward_model"]["ground_truth"]
        labels = ast.literal_eval(ground_truth)
        assert labels == original["labels"], rid
        spans += len(labels)
        quality[original["quality"]] += 1
        label_counts[str(int(bool(labels)))] += 1
        task_counts[info["task_type"]] += 1
        result = default_compute_score(
            SOURCES[split], '{"hallucination list": []}', ground_truth, info
        )
        # This is an API probe using an empty prediction on every row.
        expected_f1 = int(not labels)
        expected_penalty = 0.5 if split == "train" else 0
        assert result["format_error"] == 0 and result["span_f1"] == expected_f1, rid
        assert result["penalty"] == expected_penalty, rid
        assert result["score"] == expected_f1 - expected_penalty, rid

        try:
            tokens = tokenizer.apply_chat_template(
                row[config.prompt_key], add_generation_prompt=True, **config.apply_chat_template_kwargs
            )
        except Exception as exc:
            reasons[rid] = {"reason": "tokenizer_error", "detail": type(exc).__name__}
        else:
            lengths[rid] = len(tokens)
            if len(tokens) > config.max_prompt_length:
                reasons[rid] = {"reason": "over_max_prompt_length", "tokens": len(tokens)}

    dataset = create_rl_dataset(str(path), config, tokenizer, None, is_train=split == "train", max_samples=limit)
    kept = [row["response_id"] for row in dataset.dataframe["extra_info"]]
    assert kept == [row["id"] for row in records if row["id"] not in reasons]
    assert kept
    seen = []
    for batch in DataLoader(dataset, batch_size=32, collate_fn=collate_fn):
        shape = (len(batch["extra_info"]), config.max_prompt_length)
        assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["position_ids"].shape == shape
        token_counts = batch["attention_mask"].sum(dim=1).tolist()
        for info, raw_ids, count, route in zip(
            batch["extra_info"], batch["raw_prompt_ids"], token_counts, batch["data_source"], strict=True
        ):
            rid = info["response_id"]
            assert len(raw_ids) == count == lengths[rid], rid
            assert route == SOURCES[split]
            seen.append(rid)
    assert seen == kept

    return {
        "input": len(records),
        "retained": len(kept),
        "dropped": len(reasons),
        "dropped_by_reason": {
            name: sum(entry["reason"] == name for entry in reasons.values())
            for name in ("over_max_prompt_length", "tokenizer_error")
        },
        "dropped_response_ids": reasons,
        "max_prompt_tokens_all_tokenized": max(lengths.values()),
        "max_prompt_tokens_retained": max(lengths[rid] for rid in kept),
        "source_count": len({record["source_id"] for record in records}),
        "span_count": spans,
        "quality_counts": dict(quality),
        "binary_label_counts": dict(label_counts),
        "task_counts": dict(task_counts),
        "reward_api_checked": len(records),
        "materialized_and_collated": len(seen),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Hydra config with the launcher overrides")
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="Local tokenizer assets")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    assert cfg.data.filter_overlong_prompts and cfg.data.truncation == "error"
    assert cfg.data.reward_fn_key == "data_source" and cfg.reward_model.reward_manager == "naive"
    assert cfg.data.custom_cls.path is None and cfg.data.datagen.path is None
    tokenizer = hf_tokenizer(str(args.tokenizer_dir), trust_remote_code=False, local_files_only=True)
    sources = {row["source_id"]: row for row in read_jsonl(args.raw / "source_info.jsonl")}
    records = {split: [] for split in SOURCES}
    for row in read_jsonl(args.raw / "response.jsonl"):
        records[row["split"]].append(row)
    source_ids = {split: {row["source_id"] for row in rows} for split, rows in records.items()}
    assert not source_ids["train"] & source_ids["test"]

    report = {
        "tokenizer_dir": str(args.tokenizer_dir),
        "tokenizer_class": type(tokenizer).__name__,
        "max_prompt_length": cfg.data.max_prompt_length,
        "train_batch_size": cfg.data.get("gen_batch_size", cfg.data.train_batch_size),
        "splits": {},
    }
    for split, split_records in records.items():
        result = verify_split(split, split_records, sources, cfg.data, tokenizer)
        report["splits"][split] = result
        print(f"{split}: {json.dumps(result, ensure_ascii=False, sort_keys=True)}", flush=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
