
cd /mnt/workspace/visionllm/test/MLMCD/
seed=$1
for type in "popular" "adversarial" "random"; do
  python ./llava/eval/object_hallucination_vqa_llava.py \
  --model-path ./checkpoints/llava-v1.5-7b \
  --question-file ./POPE/output/seem/aokvqa/aokvqa_pope_seem_${type}.jsonl \
  --image-folder ./playground/data/val2014 \
  --answers-file ./answer_files_POPE_sample/llava15_aokvqa_pope_${type}_answers_no_cd_seed${seed}.jsonl --seed ${seed}
done

for type in "popular" "adversarial" "random"; do
  for ND in 999; do
    for CD_BETA in 1.0; do
      python llava/eval/object_hallucination_vqa_llava.py \
      --model-path ./checkpoints/llava-v1.5-7b \
      --question-file ./POPE/output/seem/aokvqa/aokvqa_pope_seem_${type}.jsonl \
      --answers-file ./answer_files_POPE_sample/llava15_aokvqa_pope_${type}_noisy${ND}_beta${CD_BETA}_answers${seed}.jsonl --seed ${seed} \
      --use_cd --noise_delta $ND --cd_beta $CD_BETA \
      --image-folder=./playground/data/val2014
    done
  done
done
