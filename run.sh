set -x
save_name=$1
model_name=Qwen2.5-7B-Instruct
DATA_DIR=data
model_path=Qwen/${model_name}
train_files=${DATA_DIR}/ragtruth/train.parquet
test_files=${DATA_DIR}/ragtruth/test.parquet
train_batch_size=64
max_prompt_length=16384
max_response_length=3072
lr=1e-6
use_kl_loss=True
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=4
rollout_n=16
rollout_max_num_batched_tokens=$((max_prompt_length+max_response_length))
temperature=1
topp=1
topk=-1
epoch=3

# ===== recover from the latest checkpoint =====
if [ -z "$2" ]; then
    timestamp=$(date +%m-%d-%H-%M)
else
    timestamp=$2
fi

save_path=ckpt/${timestamp}-${model_name}-${save_name}
mkdir -p $save_path
tensor_model_parallel_size=2
gpu_memory_utilization=0.6
n_gpus_per_node=8
experiment_name=${timestamp}-${model_name}-${save_name}
fsdp_config=False
rollout_data_dir=${save_path}/train_results
validation_data_dir=${save_path}/val_results
########## end of important parameters ##########

HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$fsdp_config \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$fsdp_config \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_local_dir=$save_path \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=50 \
    trainer.project_name='verl_grpo_ragtruth' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$validation_data_dir \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$topp \
    actor_rollout_ref.rollout.top_k=$topk \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=$epoch 2>&1 | tee $save_path/outputs.log