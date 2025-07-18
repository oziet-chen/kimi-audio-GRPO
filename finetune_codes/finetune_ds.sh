#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_NAME="MAP-SFT_TEST1" # HQ modify

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=hua_qiang_work-hq
export WANDB_PROJECT=kimi-audio-sft
export WANDB_API_KEY="34d6b82e63af3a14d2a380d13e390afc187876e2"
export WANDB_RUN_NAME=Kimi-Audio-SFT-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

#MODEL="moonshotai/Kimi-Audio-7B" # Set the path if you do not want to load from huggingface directly
MODEL="/root/autodl-tmp/models/Kimi-Audio-7B"

PRETRAINED_MODEL_PATH="/root/autodl-tmp/models/Kimi-Audio-7B"

# OUTPUT="/root/autodl-tmp/dataset/90/output"
OUTPUT="/root/autodl-tmp/dataset/output/$WANDB_NAME"

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
# 训练数据jsonl路径
# DATA="/root/autodl-tmp/dataset/test/audio_understanding/data_with_semantic_codes.jsonl"
# DATA="/root/autodl-tmp/dataset/train_km_semantic_codes.jsonl"
# DATA="/root/autodl-tmp/dataset/90/data_semantic_codes.jsonl"
DATA="/root/autodl-tmp/dataset/start/material/train60_km_semantic_codes.jsonl"

function usage() {
    echo '
Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model_path )
            shift
            PRETRAINED_MODEL_PATH=$1
            MODEL=$1
            ;;
        -d | --data )
            shift
            echo "$1"
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# check if wandb_name exists
if [ ! -n "$WANDB_NAME" ]; then
    echo "Error: WANDB_NAME does not exist"
    exit 1
fi
# check if data exists
if [ ! -f "$DATA" ]; then
    echo "Error: DATA file does not exist"
    exit 1
fi

# check if model_path exists
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist"
    exit 1
fi


echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "DATA: $DATA"
echo "WANDN_NAME: $WANDB_NAME"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "start finetune"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --model_path $PRETRAINED_MODEL_PATH \
    --data_path $DATA \
    --eval_ratio 0 \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name "$WANDB_NAME" \
    --deepspeed finetune_codes/ds_config_zero3.json