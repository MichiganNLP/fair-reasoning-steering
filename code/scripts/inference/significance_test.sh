#!/bin/bash

# MODEL=qwen2.5-7b-instruct
BASE_DIR="<REPO_ROOT>/results/original.unbiased_messages/${MODEL}"
DATASETS=("bbq" "crowdspairs" "genMO" "stereoset" "winoqueer")

for dataset in "${DATASETS[@]}"; do
    echo "=============================="
    echo "Running significance tests for $dataset"
    echo "=============================="

    # plain="${BASE_DIR}/${dataset}.original.${MODEL}.plain.jsonl"
    # plain="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-72b-instruct/${dataset}.original.qwen2.5-72b-instruct.plain.jsonl"
    # shift_reason="${BASE_DIR}/${dataset}.original.${MODEL}.shift_reason.phrase-0.found-by-qwen2.5-7b.run-2.jsonl"
    # phrase1="${BASE_DIR}/${dataset}.original.${MODEL}.shift_reason.phrase-1.jsonl"

    # 1. plain vs shift_reason
    # echo "[1] plain vs shift_reason"
    # 114
    model_a_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/${dataset}.original.qwen2.5-7b-instruct.shift_reason.ablation.phrase-69.jsonl"
    model_b_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/${dataset}.original.qwen2.5-7b-instruct.iasc.jsonl"

    python -m src.inference.significance_test \
        --model_a_path "$model_a_path" \
        --model_b_path "$model_b_path" \
        --dataset "$dataset"

    # # 2. plain vs phrase-1
    # echo "[2] plain vs phrase-1"
    # python -m src.inference.significance_test \
    #     --model_a_path "$plain" \
    #     --model_b_path "$phrase1" \
    #     --dataset "$dataset"

    # # 3. shift_reason vs phrase-1
    # echo "[3] shift_reason vs phrase-1"
    # python -m src.inference.significance_test \
    #     --model_a_path "$shift_reason" \
    #     --model_b_path "$phrase1" \
    #     --dataset "$dataset"

done
