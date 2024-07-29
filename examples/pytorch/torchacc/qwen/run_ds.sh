set -ex

echo "Running a deepspeed job ..."

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
FSDP_CONFIG="qwen_fsdp_ds.json"
JOB_NAME="QWEN_FSDP_DEEPSPEED_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ../../language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name Salesforce/wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --preprocessing_num_workers 90 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir ./ds_outputs \
    --overwrite_output_dir \
    --config_name ./Qwen2-7B/ \
    --tokenizer_name ./Qwen2-7B/ \
    --trust_remote_code true \
    --low_cpu_mem_usage true \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --cache_dir ../cache \
    --save_strategy epoch \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --$PRECISION \
    --deepspeed $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
