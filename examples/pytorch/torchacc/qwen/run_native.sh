set -ex

echo "Running a native torch job ..."

export USE_TORCH_XLA=0

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010
[ -z "$TASK_TAG" ] && TASK_TAG=0000
[ -z "$BS" ] && BS=1
[ -z "$SEQLEN" ] && SEQLEN=4096


NPROC_PER_NODE=8
PRECISION="bf16=true"
FSDP_CONFIG="qwen_fsdp_native.json"
JOB_NAME="QWEN_FSDP_NATIVE_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ../../language-modeling/run_clm.py \
    --num_train_epochs 1 \
    --dataset_name Salesforce/wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --preprocessing_num_workers 90 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir ./native_outputs2 \
    --overwrite_output_dir \
    --model_name_or_path ./Qwen2-7B/ \
    --tokenizer_name ./Qwen2-7B/ \
    --trust_remote_code true \
    --low_cpu_mem_usage true \
    --cache_dir ../cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy epoch \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --$PRECISION \
    --fsdp "auto_wrap" \
    --fsdp_config $FSDP_CONFIG 2>&1 | tee ./${JOB_NAME}_${RANK}_${TASK_TAG}.log
