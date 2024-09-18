set -ex
#3env -i bash

#cd /root/shw/pytorch/xla
#TORCH_CUDA_ARCH_LIST="8.0" TF_CUDA_COMPUTE_CAPABILITIES="8.0" XLA_CUDA=1 DEBUG=0 python setup.py develop
#cd /root/shw/test_ga

echo "Running a torch job with torchacc ..."

export XLA_THREAD_POOL_SIZE=1
export PJRT_ALLOCATOR_FRACTION=0.97
export PJRT_DEVICE=CUDA
export XLA_FLAGS='--xla_gpu_memory_limit_slop_factor=500'
#export XLA_PERSISTENT_CACHE_PATH=./compiled_cache # uncomment this line to cache the compile results and speed up initialization.

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9007

#export CUDA_VISIBLE_DEVICES=4,5,6,7

BS=1
SEQLEN=4096
NPROC_PER_NODE=4
PRECISION="bf16=true"
FSDP_CONFIG="../examples/pytorch/torchacc/llama3/llama3_fsdp_native.json"
JOB_NAME="LLAMA3_FSDP_TORCHACC_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA"
#export XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1
#export XLA_HLO_DEBUG_VERBOSE_STACK=1 USE_TORCHACC=1 XLA_DUMP_FATAL_STACK=1 XLA_DUMP_HLO_GRAPH=1 PT_XLA_DEBUG=1 XLA_HLO_DEBUG=1
#export XLA_SAVE_TENSORS_FILE=XLA_SAVE_TENSORS_FILE.txt XLA_SAVE_HLO_FILE=XLA_SAVE_HLO_FILE.txt XLA_SAVE_TENSORS_FMT=hlo XLA_METRICS_FILE=XLA_METRICS_FILE.txt
#rm -rf ./hlo_normal_new
#export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_dump_to=./hlo_normal_new"

#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_VMODULE=gpu_hlo_schedule=10
#export TF_CPP_VMODULE=bfc_allocator=2
#export TF_CPP_VMODULE=xla_graph_executor=10
#rm -rf ./log/acc_bs1_ga2
#rm -rf ./log/torchrun/
#--log-dir ./log/torchrun \
#        -r 3 \
#        -t 3 \
#export CHUNK_SIZE=218210304

#--resume_from_checkpoint /root/shw/test_save_checkpoint/ckpt_xla_old/checkpoint-10 \
torchrun    --nproc_per_node $NPROC_PER_NODE \
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
    --output_dir ./xla_ckpt/ \
    --overwrite_output_dir \
    --config_name ../examples/pytorch/torchacc/llama3/Meta-Llama-3-8B/ \
    --tokenizer_name ../examples/pytorch/torchacc/llama3/Meta-Llama-3-8B/ \
    --trust_remote_code true \
    --low_cpu_mem_usage true \
    --cache_dir ../cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy steps \
	--save_steps 100 \
    --logging_strategy steps \
    --logging_steps 1 \
    --gradient_checkpointing no \
    --gradient_accumulation 1 \
    --logging_dir ./log/test_acc_save_ckpt/test_new/torhacc_gpu4_2_resume \
    --$PRECISION \
    --fsdp "auto_wrap" \
    --fsdp_config $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
    #--resume_from_checkpoint /root/shw/test_save_checkpoint/ckpt_xla_old/checkpoint-10 \
    #--save_strategy steps \
	#--save_steps 100 \

