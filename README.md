# RLSeek: Evidence-Grounded Reasoning for RAG Hallucination Detection

ACL 2026

code directory:

- verl/: official verl implementation for reinforcement learning, with our code provided in it.
	- verl/verl/utils/reward_score: init span-f1 reward and penalty item.
	- verl/verl/trainer/ppo/core_algos.py: init CAPO method.
- run.sh: script to run the code.


## Get Started


### Step 1. Install dependencies

```shell
conda create -n rlseek python==3.10
conda activate rlseek
pip install torch packaging ninja
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash-attn==2.8.0
pip install vllm==0.8.5
pip install onnxruntime-gpu datasets transformers python-dotenv uuid pandas swanlab openai tqdm jsonlines tiktoken
pip install flashinfer-python
cd verl
pip install -e . --user
cd ..
```


### Step 2. Run the script to reproduce the experimental results

To start training:
```shell
bash run.sh run_grpo_7B
```

To recover from the latest checkpoint and continue training, run:
```shell
bash run.sh <save_name> <last_ckpt_name>
```