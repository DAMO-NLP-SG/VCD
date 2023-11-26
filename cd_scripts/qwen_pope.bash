
cd /mnt/workspace/visionllm/test/MLMCD/
mkdir answer_files_POPE_qwen_stage2_sample
dataset=$1
seed=$2

# coco gqa aokvqa

# for type in "popular" "adversarial" "random"; do
#   python ./llava/eval/object_hallucination_vqa_qwenvl.py \
#   --question-file ./POPE/output/coco/coco_pope_${type}.json \
#   --image-folder ./playground/data/val2014 \
#   --answers-file ./answer_files_POPE_qwen_stage2_sample/qwen_coco_pope_${type}_answers_no_cd_seed${seed}.jsonl --seed ${seed}
# done

# for type in "popular" "adversarial" "random"; do
#   python ./llava/eval/object_hallucination_vqa_qwenvl.py \
#   --question-file ./POPE/output/seem/gqa/gqa_pope_seem_${type}.jsonl \
#   --image-folder /mnt/workspace/visionllm/data/llava_tuning/gqa/images \
#   --answers-file ./answer_files_POPE_qwen_stage2_sample/qwen_gqa_pope_${type}_answers_no_cd_seed${seed}.jsonl --seed ${seed}
# done


# for type in "popular" "adversarial" "random"; do
#   python ./llava/eval/object_hallucination_vqa_qwenvl.py \
#   --question-file ./POPE/output/seem/aokvqa/aokvqa_pope_seem_${type}.jsonl \
#   --image-folder ./playground/data/val2014 \
#   --answers-file ./answer_files_POPE_qwen_stage2_sample/qwen_aokvqa_pope_${type}_answers_no_cd_seed${seed}.jsonl --seed ${seed}
# done

if [ "$dataset" == 'coco' ]; then
  for type in "popular" "adversarial" "random"; do
    for ND in 999; do
      for CD_BETA in 1.0; do
        python llava/eval/object_hallucination_vqa_qwenvl.py \
        --question-file ./POPE/output/coco/coco_pope_${type}.json \
        --answers-file ./answer_files_POPE_qwen_stage2_sample/qwen_coco_pope_${type}_noisy${ND}_beta${CD_BETA}_answers${seed}.jsonl --seed ${seed} \
        --use_cd --noise_delta $ND --cd_beta $CD_BETA \
        --image-folder=./playground/data/val2014
      done
    done
  done
fi


if [ "$dataset" == 'gqa' ]; then
  for type in "popular" "adversarial" "random"; do
    for ND in 999; do
      for CD_BETA in 1.0; do
        python llava/eval/object_hallucination_vqa_qwenvl.py \
        --question-file ./POPE/output/seem/gqa/gqa_pope_seem_${type}.jsonl \
        --answers-file ./answer_files_POPE_qwen_stage2_sample/qwen_gqa_pope_${type}_noisy${ND}_beta${CD_BETA}_answers${seed}.jsonl --seed ${seed} \
        --use_cd --noise_delta $ND --cd_beta $CD_BETA \
        --image-folder /mnt/workspace/visionllm/data/llava_tuning/gqa/images
      done
    done
  done
fi

if [ "$dataset" == 'aokvqa' ]; then
  for type in "popular" "adversarial" "random"; do
    for ND in 999; do
      for CD_BETA in 1.0; do
        python llava/eval/object_hallucination_vqa_qwenvl.py \
        --question-file ./POPE/output/seem/aokvqa/aokvqa_pope_seem_${type}.jsonl \
        --answers-file ./answer_files_POPE_qwen_stage2_sample/qwen_aokvqa_pope_${type}_noisy${ND}_beta${CD_BETA}_answers${seed}.jsonl --seed ${seed} \
        --use_cd --noise_delta $ND --cd_beta $CD_BETA \
        --image-folder ./playground/data/val2014
      done
    done
  done
fi
# for type in "popular" "adversarial" "random"; do
#   for ND in 999; do
#     for CD_BETA in 1.0; do
#       python llava/eval/object_hallucination_vqa_minigpt4_v2.py \
#       --question-file ./POPE/output/coco/coco_pope_${type}.json \
#       --answers-file ./answer_files_POPE_minigpt4_v2_stage2_sample/minigpt4v2_coco_pope_${type}_noisy${ND}_beta${CD_BETA}_answers${seed}.jsonl --seed ${seed} \
#       --use_cd --noise_delta $ND --cd_beta $CD_BETA \
#       --image-folder=./playground/data/val2014
#     done
#   done
# done


