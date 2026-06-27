#!/bin/bash
set -euo pipefail

gpu=0

model_paths=(
  "<HF_HOME>/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
  # "<HF_HOME>/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)

# DATA_IDX=${DATA_IDX:?DATA_IDX not set}
DATA_IDX=0


data_paths=(
  "<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/crowdspairs/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/genMO/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/stereoset/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/winoqueer/processed/test/test.jsonl"
)

for model_path in "${model_paths[@]}"; do
  # extract model short name
  # model_dir="$(dirname "$(dirname "$model_path")")"
  # model_short="$(basename "$model_dir")"
  # model_short="$(echo "$model_short" | sed 's/^models--Qwen--//; s/^models--meta-llama--//')"
  # model_short="$(echo "$model_short" | tr '[:upper:]' '[:lower:]')"
  model_short="qwen2.5-7b-instruct"

  for jsonfilename in "${data_paths[@]}"; do
  # jsonfilename="${data_paths[$DATA_IDX]}"

  # extract dataset name
  dataset=$(basename "$(dirname "$(dirname "$(dirname "$jsonfilename")")")")

  pred_filepath="<REPO_ROOT>/results/original.unbiased_messages/${model_short}/${dataset}.original.${model_short}.plain.jsonl"

  echo "Running Luo Method"
  echo "  model:    $model_short"
  echo "  dataset:  $dataset"
  echo "  predfile: $pred_filepath"

  output_filepath=<REPO_ROOT>/results/original.unbiased_messages/${model_short}/${dataset}.original.${model_short}.self_debias_reprompting.enhanced.jsonl

  echo "  output file: $output_filepath"

  python -m src.other_methods.self_debiasing_via_reprompting \
    --jsonfilename $pred_filepath \
    --outputfilename $output_filepath \
    --modelid $model_path \
    --gpu 0
  done
done