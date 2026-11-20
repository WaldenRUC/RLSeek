#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
    printf 'Usage: RLSEEK_PYTHON=/path/to/python %s REPO_ROOT NEW_BUILD_DIR\n' "$0" >&2
    exit 2
fi
tools_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$1" && pwd)"
if [[ -n "${RLSEEK_TOKENIZER_DIR:-}" ]]; then
    tokenizer_dir="$(cd "$RLSEEK_TOKENIZER_DIR" && pwd)"
fi
mkdir "$2"
build="$(cd "$2" && pwd)"
mkdir "$build/raw"
setup_dir="${RLSEEK_SETUP_DIR:-/workspace/.rlseek}"
python="${RLSEEK_PYTHON:-$setup_dir/venv/bin/python}"
export PYTHONPATH="$repo/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_DATASETS_CACHE="$build/loader-cache"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$setup_dir/tiktoken}"
revision=c103204b9ce28d6bbad859304bf30de72b8ed8fe
for file in response.jsonl source_info.jsonl; do
    curl -fsSL "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/$revision/dataset/$file" -o "$build/raw/$file"
done
cd "$build"
grep '  raw/' "$tools_dir/reports/data.sha256" | sha256sum --check
"$python" -u "$tools_dir/prepare_ragtruth.py" --raw raw --output full | tee conversion.log
grep '  full/' "$tools_dir/reports/data.sha256" | sha256sum --check | tee reproducibility.log

if [[ -z "${RLSEEK_TOKENIZER_DIR:-}" ]]; then
    tokenizer_dir="$build/tokenizer-qwen25-npm"
    mkdir "$tokenizer_dir"
    tarball="$build/tokenizer.tgz"
    curl -fsSL 'https://registry.npmjs.org/@lenml/tokenizer-qwen2_5/-/tokenizer-qwen2_5-3.7.2.tgz' -o "$tarball"
    "$python" - "$tarball" "$tokenizer_dir" <<'PY'
import base64
import hashlib
import sys
import tarfile
from pathlib import Path

bundle, out = map(Path, sys.argv[1:])
expected = "6hbFsuKfOU7pncR3qQqvhOAcDboqWTZOTob9jv4sJZjXhc8X05r4pretlPeyC2UeLU/ayFDU6rl5dd/BDDUsFg=="
if base64.b64encode(hashlib.sha512(bundle.read_bytes()).digest()).decode() != expected:
    raise SystemExit("Tokenizer tarball SHA-512 mismatch")
with tarfile.open(bundle) as archive:
    for name in ("tokenizer.json", "tokenizer_config.json"):
        with archive.extractfile(f"package/models/{name}") as member:
            (out / name).write_bytes(member.read())
PY
    sha256sum --check "$tools_dir/reports/tokenizer.sha256"
fi

cd "$repo"
if [[ -e data/ragtruth || -L data/ragtruth ]]; then
    cmp "$build/full/train.parquet" data/ragtruth/train.parquet
    cmp "$build/full/test.parquet" data/ragtruth/test.parquet
else
    mkdir -p data
    ln -s "$build/full" data/ragtruth
fi
"$python" -u "$tools_dir/verify_full.py" --raw "$build/raw" \
    --config "$tools_dir/loader-config.yaml" \
    --tokenizer-dir "$tokenizer_dir" --report "$build/verification.json" 2>&1 | tee "$build/verification.log"
