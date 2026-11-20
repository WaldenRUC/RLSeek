"""Exercise RLSeek's actual loader with an offline byte-level smoke tokenizer."""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from verl.trainer.main_ppo import create_rl_dataset
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.reward_score import default_compute_score

from prepare_ragtruth import SCHEMA, SOURCES, read_jsonl, reference_text


def local_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: idx for idx, token in enumerate(["<pad>", "<unk>", *alphabet])}
    backend = Tokenizer(models.BPE(vocab, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        chat_template=(
            "{% for item in messages %}{{ item['role'] + '\\n' + item['content'] + '\\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ 'assistant\\n' }}{% endif %}"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    originals = {row["id"]: row for row in read_jsonl(args.raw / "response.jsonl")}
    sources = {row["source_id"]: row for row in read_jsonl(args.raw / "source_info.jsonl")}
    tokenizer = local_tokenizer()
    config = OmegaConf.create({
        "prompt_key": "prompt",
        "max_prompt_length": 16384,
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": 1,
        "truncation": "error",
        "return_raw_chat": True,
    })
    for split, route in SOURCES.items():
        path = args.dataset / f"{split}.parquet"
        table = pq.read_table(path)
        assert table.schema.equals(SCHEMA, check_metadata=False), table.schema
        assert table.schema.metadata[b"provenance"] == SCHEMA.metadata[b"provenance"]
        dataset = create_rl_dataset(str(path), config, tokenizer, None, is_train=(split == "train"))
        assert len(dataset) == table.num_rows > 0
        counts = Counter()
        reward_results = Counter()
        max_tokens = 0
        for batch in DataLoader(dataset, batch_size=4, collate_fn=collate_fn):
            assert batch["input_ids"].shape[1] == config.max_prompt_length
            assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["position_ids"].shape
            max_tokens = max(max_tokens, int(batch["attention_mask"].sum(dim=1).max()))
            for extra, reward, data_source, messages in zip(
                batch["extra_info"], batch["reward_model"], batch["data_source"], batch["raw_prompt"],
                strict=True,
            ):
                original = originals[extra["response_id"]]
                source = sources[original["source_id"]]
                labels = ast.literal_eval(reward["ground_truth"])
                assert labels == original["labels"]
                assert original["split"] == split and extra["source_id"] == original["source_id"]
                assert data_source == route and extra["binary_label"] == int(bool(labels))
                assert extra["response"] == original["response"] and extra["reference"] == reference_text(source)
                assert messages[0]["role"] == "user"
                assert messages[0]["content"].endswith(extra["response"])
                counts[(extra["task_type"], extra["binary_label"])] += 1
                # Exercise the reward API with annotation-derived predictions, not a model.
                quote = next(line.strip() for line in extra["reference"].splitlines() if line.strip())
                solution = "## Step 1\n<quote>" + quote + "</quote>\n" + json.dumps(
                    {"hallucination list": [label["text"] for label in labels]}, ensure_ascii=False
                )
                result = default_compute_score(route, solution, reward["ground_truth"], extra)
                assert result["format_error"] == 0 and 0 <= result["span_f1"] <= 1
                assert result["penalty"] == 0
                reward_results["F1=1" if result["span_f1"] == 1 else "F1<1"] += 1
        assert sum(counts.values()) == len(dataset)
        print(f"{split}: verified {len(dataset)} rows, max prompt tokens={max_tokens}")
        print("Groups:", sorted(counts.items()), "Reward smoke:", dict(reward_results))


if __name__ == "__main__":
    main()
