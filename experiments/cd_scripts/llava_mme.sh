#!/bin/bash

seed=${1:-200}
beta=${2:-1.0}
noise_delta=${3:-500}
cd_alpha=${4:-0.1}
cd /mnt/workspace/visionllm/test/MLMCD
python ./llava/eval/MME_llava.py \
    --model-path /mnt/workspace/workgroup/zh/llava-v1.5-7b \
    --question-file ./playground/data/MME/llava_mme.jsonl \
    --image-folder ./playground/data/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/MME/answers/llava-v1.5-7b-$seed-fullresult-${noise_delta}-${beta}-${cd_alpha}.jsonl \
    --conv-mode vicuna_v1 \
    --noise_delta $noise_delta \
    --cd_beta $beta \
    --cd_alpha $cd_alpha \
    --seed $seed \
    --use_cd
cd ./playground/data/MME
python convert_answer_to_mme.py --experiment llava-v1.5-7b-$seed-fullresult-${noise_delta}-${beta}-${cd_alpha}


# cd /mnt/workspace/visionllm/test/MLMCD/playground/data/MME/eval_tool
# python calculation.py --results_dir llava-v1.5-7b-$seed-fullresult-${noise_delta}-${beta}