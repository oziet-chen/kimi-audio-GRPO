#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

## grpo
export DEBUG_MODE="true" 
export LOG_PATH="./debug_log_avqa-r1-MAP-GRPO_TEST1.txt"
export WANDB_NAME="MAP-GRPO_TEST6" 
export CUDA_LAUNCH_BLOCKING=1

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=hua_qiang_work-hq
export WANDB_PROJECT=kimi-audio-grpo
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

## to config
#MODEL="moonshotai/Kimi-Audio-7B" # Set the path if you do not want to load from huggingface directly
# MODEL="/root/autodl-tmp/models/Kimi-Audio-7B"
MODEL="/root/autodl-tmp/models/Kimi-Audio-7B-Instruct"

# PRETRAINED_MODEL_PATH="/root/autodl-tmp/models/Kimi-Audio-7B"
PRETRAINED_MODEL_PATH="/root/autodl-tmp/models/Kimi-Audio-7B-Instruct"
# PRETRAINED_MODEL_PATH="/root/autodl-tmp/dataset/output/Kimi-Audio-7B-SFT-MAP1-After"

# OUTPUT="/root/autodl-tmp/dataset/90/output"
OUTPUT="/root/autodl-tmp/dataset/output/$WANDB_NAME"

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
# 训练数据jsonl路径
# DATA="/root/autodl-tmp/dataset/test/audio_understanding/data_with_semantic_codes.jsonl"
# DATA="/root/autodl-tmp/dataset/train_km_semantic_codes.jsonl"
# DATA="/root/autodl-tmp/dataset/90/data_semantic_codes.jsonl"
# DATA="/root/autodl-tmp/dataset/start/material/val60_km_test_grpo_semantic_codes.jsonl"
# DATA="/root/autodl-tmp/dataset/start/material/val60_km_test_grponothink_semantic_codes.jsonl"
DATA="/root/autodl-tmp/dataset/start/material/val60_km_val_grponothink_semantic_codes.jsonl"


# CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.extract_semantic_codes --model_name "/root/autodl-tmp/models/Kimi-Audio-7B" --input_file "/root/autodl-tmp/dataset/start/material/val60_km_test_grpo.jsonl" --output_file "/root/autodl-tmp/dataset/start/material/val60_km_test_grpo_semantic_codes.jsonl"


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

echo "start grpo"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"



##
torchrun $DISTRIBUTED_ARGS grpo.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATA \
    --model_max_length 8192 \
    --max_completion_length 256 \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --len_control false \
    --weighted_reward false\
    --beta 0.001 \
    --max_grad_norm 5 \
    --save_only_model true \
    --attn_implementation flash_attention_2 \
    --model_type kimi \
    --logging_steps 1 \
    --report_to "wandb" \
    --num_generations 8 \
    --gradient_checkpointing True \
    --run_name "$WANDB_NAME" \
    --deepspeed finetune_codes/ds_config_zero3.json

