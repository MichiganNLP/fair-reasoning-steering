#!/bin/bash
set -euo pipefail

gpu=0

model_paths=(
  "<HF_HOME>/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
  # "<HF_HOME>/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)

DATA_IDX=${DATA_IDX:?DATA_IDX not set}


data_paths=(
  "<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/crowdspairs/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/genMO/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/stereoset/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/winoqueer/processed/test/test.jsonl"
)

for model_path in "${model_paths[@]}"; do
  # extract model short name
  model_dir="$(dirname "$(dirname "$model_path")")"
  model_short="$(basename "$model_dir")"
  model_short="$(echo "$model_short" | sed 's/^models--Qwen--//; s/^models--meta-llama--//')"
  model_short="$(echo "$model_short" | tr '[:upper:]' '[:lower:]')"


  # for jsonfilename in "${data_paths[@]}"; do
  jsonfilename="${data_paths[$DATA_IDX]}"

  # extract dataset name
  dataset=$(basename "$(dirname "$(dirname "$(dirname "$jsonfilename")")")")

  prediction_file="<REPO_ROOT>/results/original.unbiased_messages/${model_short}/${dataset}.original.${model_short}.plain.jsonl"

  echo "Running ADBP"
  echo "  model:    $model_short"
  echo "  dataset:  $dataset"
  echo "  predfile: $prediction_file"

  outputfilename=<REPO_ROOT>/results/original.unbiased_messages/${model_short}/${dataset}.original.${model_short}.adbp.jsonl

  echo "  output file: $outputfilename"

  python -m src.other_methods.adbp \
    "$jsonfilename" \
    "$outputfilename" \
    "$model_path" \
    "$gpu" \
    "$prediction_file"

  # done
done