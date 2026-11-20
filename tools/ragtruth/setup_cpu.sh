#!/usr/bin/env bash
set -euo pipefail
setup_dir="${RLSEEK_SETUP_DIR:-/workspace/.rlseek}"
mkdir -p "$setup_dir"
setup_dir="$(cd "$setup_dir" && pwd)"
export UV_CACHE_DIR="$setup_dir/cache"
export UV_PYTHON_INSTALL_DIR="$setup_dir/python"
export UV_SYSTEM_CERTS=true
uv python install --no-bin 3.10.22
uv venv --allow-existing --python 3.10.22 "$setup_dir/venv"
python="$setup_dir/venv/bin/python"
uv pip install --python "$python" --index https://download.pytorch.org/whl/cpu 'torch==2.6.0' 'torchvision==0.21.0'
export PYTHONDONTWRITEBYTECODE=1
uv pip install --python "$python" \
  'torch==2.6.0+cpu' 'torchvision==0.21.0+cpu' \
  'transformers==4.51.3' 'peft==0.15.2' 'tensordict==0.8.3' 'torchdata==0.11.0' \
  'accelerate==1.6.0' 'ray[default]==2.49.2' \
  'numpy==1.26.4' 'pyarrow==25.0.1' 'tokenizers==0.21.4' codetiming 'datasets==3.6.0' dill 'hydra-core==1.3.8' 'omegaconf==2.3.1' pandas pybind11 pylatexenc \
  wandb 'packaging>=20' tensorboard 'tiktoken==0.14.0' uvicorn fastapi latex2sympy2_extended math_verify
uv pip check --python "$python"
# Populate the standard tiktoken cache with the original, hash-verified data.
wheel="$(mktemp)"
trap 'rm -f "$wheel"' EXIT
curl -fsSL https://files.pythonhosted.org/packages/02/b7/350ff16dfb0715db4a61823a375461149a3c0169f31b873b7c18bbbc3938/tiktoken_offline-0.1.1-py3-none-any.whl -o "$wheel"
printf '%s  %s\n' d1bb6dbc1dbf2b71ebd483e6747d10d8f91a29521075cee01377448c699f7cf0 "$wheel" | sha256sum --check --status
export TIKTOKEN_CACHE_DIR="$setup_dir/tiktoken"
"$python" - "$wheel" <<'PYTHON'
import hashlib
import os
import sys
from pathlib import Path
from zipfile import ZipFile

url = b"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
with ZipFile(sys.argv[1]) as wheel:
    data = wheel.read("tiktoken_ext/data/cl100k_base.tiktoken")
if hashlib.sha256(data).hexdigest() != "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7":
    raise SystemExit("Invalid cl100k_base SHA-256")
cache = Path(os.environ["TIKTOKEN_CACHE_DIR"])
cache.mkdir(parents=True, exist_ok=True)
(cache / hashlib.sha1(url).hexdigest()).write_bytes(data)
PYTHON
"$python" -c 'import tiktoken; assert tiktoken.get_encoding("cl100k_base").encode("hello") == [15339]'
