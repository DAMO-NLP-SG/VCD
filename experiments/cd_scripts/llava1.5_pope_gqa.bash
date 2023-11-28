
cd /mnt/workspace/visionllm/test/MLMCD/
seed=$1
for type in "popular" "adversarial" "random"; do
  python ./llava/eval/object_hallucination_vqa_llava.py \
  --model-path ./checkpoints/llava-v1.5-7b \
  --question-file ./POPE/output/seem/gqa/gqa_pope_seem_${type}.jsonl \
  --image-folder /mnt/workspace/visionllm/data/llava_tuning/gqa/images \
  --answers-file ./answer_files_POPE_sample/llava15_gqa_pope_${type}_answers_no_cd_seed${seed}.jsonl --seed ${seed}
done

for type in "popular" "adversarial" "random"; do
  for ND in 999; do
    for CD_BETA in 1.0; do
      python llava/eval/object_hallucination_vqa_llava.py \
      --model-path ./checkpoints/llava-v1.5-7b \
      --question-file ./POPE/output/seem/gqa/gqa_pope_seem_${type}.jsonl \
      --answers-file ./answer_files_POPE_sample/llava15_gqa_pope_${type}_noisy${ND}_beta${CD_BETA}_answers${seed}.jsonl --seed ${seed} \
      --use_cd --noise_delta $ND --cd_beta $CD_BETA \
      --image-folder=/mnt/workspace/visionllm/data/llava_tuning/gqa/images
    done
  done
done




# for type in "popular" "adversarial" "random"; do
#   for ND in 999; do
#     for CD_BETA in 0.5 0.8 1.0; do
#       python llava/eval/object_hallucination_vqa_llava.py \
#       --model-path ./checkpoints/llava-v1.5-7b \
#       --question-file ./POPE/output/coco/coco_pope_${type}.json \
#       --answers-file ./answer_files_POPE/llava15_coco_pope_${type}_noisy${ND}_beta${CD_BETA}_answers.jsonl \
#       --use_cd --noise_delta $ND --cd_beta $CD_BETA \
#       --image-folder=./playground/data/val2014
#     done
#   done
# done


# echo "ND    CD_BETA    Precision    Recall    F1";
# for ND in 800 500 200; do
#   for CD_BETA in 0.2 0.5 0.8 1.0; do
#     result=$(python llava/eval/eval_pope.py --gen_files answer_files_POPE/llava15_coco_pope_popular_noisy${ND}_beta${CD_BETA}_answers.jsonl --gt_files POPE/output/coco/coco_pope_popular.json)
#     precision=$(echo "$result" | grep -oP 'precision:\s+\K[0-9.]+')
#     recall=$(echo "$result" | grep -oP 'recall:\s+\K[0-9.]+')
#     f1=$(echo "$result" | grep -oP 'f1:\s+\K[0-9.]+')
#     echo "$ND    $CD_BETA    $precision    $recall    $f1"
#   done
# done