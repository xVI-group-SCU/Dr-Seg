export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=pretrained_models/Qwen2.5-VL-7B-Instruct # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=training_scripts/drseg_7b.yaml \
    data.train_files=data/VisionReasoner_multi_object_7k_840 \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.reward.compute_score=dr_seg \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=10 \
    trainer.save_checkpoint_path=your_path_to_checkpoint/${RUN_NAME}
