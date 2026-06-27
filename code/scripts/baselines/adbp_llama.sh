#!/bin/bash
set -euo pipefail

gpu=0
DATA_IDX=${DATA_IDX:?DATA_IDX not set}

model_path="<HF_HOME>/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

data_paths=(
  "<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/crowdspairs/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/genMO/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/stereoset/processed/test/test.jsonl"
  "<REPO_ROOT>/datasets/winoqueer/processed/test/test.jsonl"
)

jsonfilename="${data_paths[$DATA_IDX]}"

dataset=$(basename "$(dirname "$(dirname "$(dirname "$jsonfilename")")")")

model_short="llama3.1-8b-instruct"

prediction_file="<REPO_ROOT>/results/original.unbiased_messages/${model_short}/${dataset}.original.${model_short}.plain.jsonl"

outputfilename="<REPO_ROOT>/results/original.unbiased_messages/${model_short}/${dataset}.original.${model_short}.adbp.jsonl"

echo "Running ADBP"
echo "  DATA_IDX: $DATA_IDX"
echo "  dataset:  $dataset"
echo "  model:    $model_short"
echo "  predfile: $prediction_file"
echo "  outfile:  $outputfilename"

python -m src.other_methods.adbp \
  "$jsonfilename" \
  "$outputfilename" \
  "$model_path" \
  "$gpu" \
  "$prediction_file"
