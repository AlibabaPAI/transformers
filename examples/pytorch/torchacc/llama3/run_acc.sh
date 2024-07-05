set -ex

echo "Running a torch job with torchacc ..."

export PJRT_ALLOCATOR_FRACTION=0.90
export PJRT_DEVICE=CUDA
export XLA_FLAGS='--xla_gpu_memory_limit_slop_factor=500 --xla_multiheap_size_constraint_per_heap=15032385536 --xla_dump_hlo_as_text --xla_dump_to=./cp8_hlo'
# export LOW_CPU_MEM_USAGE=1
#export XLA_PERSISTENT_CACHE_PATH=./compiled_cache # uncomment this line to cache the compile results and speed up initialization.

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010
[ -z "$TASK_TAG" ] && TASK_TAG=0000
[ -z "$BS" ] && BS=2
[ -z "$SEQLEN" ] && SEQLEN=8192

NPROC_PER_NODE=8
PRECISION="bf16=true"
FSDP_CONFIG="llama3_fsdp_acc.json"
JOB_NAME="LLAMA3_FSDP_TORCHACC_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA_FULLGC_NOFLAT"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ../../language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --config_name ./Meta-Llama-3-70B/ \
    --tokenizer_name ./Meta-Llama-3-70B/ \
    --trust_remote_code true \
    --low_cpu_mem_usage true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --$PRECISION \
    --fsdp "full_shard" \
    --fsdp_config $FSDP_CONFIG 2>&1 | tee ./${JOB_NAME}_${RANK}_${TASK_TAG}.log
