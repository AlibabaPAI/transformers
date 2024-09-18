set -ex

echo "Running a native torch job ..."

export USE_TORCH_XLA=0

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010

#export CUDA_VISIBLE_DEVICES=4,5,6,7
BS=1
SEQLEN=4096
NPROC_PER_NODE=4
PRECISION="bf16=true"
FSDP_CONFIG="../examples/pytorch/torchacc/llama3/llama3_fsdp_native.json"
JOB_NAME="LLAMA3_FSDP_NATIVE_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA"

#--resume_from_checkpoint /root/shw/test_save_checkpoint/native_ckpt/checkpoint-10 \
torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ../examples/pytorch/language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir ./native_ckpt/ \
    --overwrite_output_dir \
    --config_name ../examples/pytorch/torchacc/llama3/Meta-Llama-3-8B/ \
    --tokenizer_name ../examples/pytorch/torchacc/llama3/Meta-Llama-3-8B/ \
    --trust_remote_code true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy steps \
    --save_steps 100 \
    --gradient_checkpointing no \
    --gradient_accumulation 1 \
    --logging_dir ./log/test_native_save_ckpt/\
    --logging_steps 1 \
    --$PRECISION \
    --fsdp "auto_wrap" \
    --fsdp_config $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
#    --save_strategy steps \
#    --save_steps 10 \