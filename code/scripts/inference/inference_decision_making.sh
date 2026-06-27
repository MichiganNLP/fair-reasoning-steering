

# export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=<project-name>

export PYTHONPATH="<REPO_ROOT>/src"


# ---------- models ----------
model_paths=(
  "<HF_HOME>/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
  # "<HF_HOME>/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  # "<HF_HOME>/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
  # "<HF_HOME>/hub/models--tiiuae--Falcon3-7B-Instruct/snapshots/1e57a0ecd176c7c139f289c60a74e57f887c3dfb"
  # "<HF_HOME>/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
  # "<HF_HOME>/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
  # "<HF_HOME>/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
  # "<HF_HOME>/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
  # "<HF_HOME>/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e"
  # "<HF_HOME>/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots/a07cc9a04f16550a088caea529712d1d335b0ac1"

  ##############################
  # sft models
  ##############################
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-gpt-5-mini/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-manual-modify/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-qwen-2.5-7b/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/qwen2.5-instruct-qwen-2.5-72b/full/sft"

  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-gpt-5-mini/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-llama-3-8b/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-manual-modify/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-qwen-2.5-7b/full/sft"
  # "<REPO_ROOT>/llamafactory.checkpoints/llama3-instruct-qwen-2.5-72b/full/sft"
)
model_tags=(
  "qwen2.5-7b-instruct"
  # "llama3.1-8b-instruct"
  # "mistral-v0.3-7b-instruct"
  # "falcon3-7b-instruct"
  # "qwen2.5-14b-instruct"
  # "qwen3-8b.no_thinking"
  # "qwen3-8b.thinking"
  # "qwen2.5-72b-instruct"
  # "llama3.1-70b-instruct"
  # "deepseek-r1-llama-70b"
  # "deepseek-r1-llama-70b.no_thinking"
  # "smollm3-4b.thinking"
  # "smollm3-4b.non_thinking"

  ##############################
  # sft models
  ##############################
  # "sft.qwen2.5-7b-instruct-gpt-5-mini"
  # "sft.qwen2.5-7b-instruct-manual-modify"
  # "sft.qwen2.5-7b-instruct-qwen2.5-7b"
  # "sft.qwen2.5-7b-instruct-qwen2.5-72b"

  # "sft.llama3-8b-instruct-gpt-5-mini"
  # "sft.llama3-8b-instruct-llama3-8b"
  # "sft.llama3-8b-instruct-manual-modify"
  # "sft.llama3-8b-instruct-qwen2.5-7b"
  # "sft.llama3-8b-instruct-qwen2.5-72b"
)

# model_path=<HF_HOME>/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
# model_tag=qwen2.5-7b
# model_path=<HF_HOME>/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
# model_tag=qwen2.5-14b-instruct.trial

data_paths=(
  ############################################
  # Initial sub-sample experiments
  ############################################

  # "<REPO_ROOT>/datasets/bbq/processed/BBQ.json"
  # "<REPO_ROOT>/datasets/crowdspairs/processed/data.jsonl"
  # "<REPO_ROOT>/datasets/genMO/processed/data.jsonl"
  # "<REPO_ROOT>/datasets/stereoset/processed/data.jsonl"
  # "<REPO_ROOT>/datasets/winoqueer/processed/data.jsonl"

  ############################################
  # Generating ICL/SFT examples
  ############################################
  # "<REPO_ROOT>/datasets/bbq/processed/train/train.100.random.jsonl"

  ############################################
  # Processed test
  ############################################
  # "<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl"
  # "<REPO_ROOT>/datasets/crowdspairs/processed/test/test.jsonl"
  # "<REPO_ROOT>/datasets/genMO/processed/test/test.jsonl"
  # "<REPO_ROOT>/datasets/stereoset/processed/test/test.jsonl"
  # "<REPO_ROOT>/datasets/winoqueer/processed/test/test.jsonl"

  "<REPO_ROOT>/datasets/biobias/processed/test.subset.jsonl"
  # "<REPO_ROOT>/datasets/discrim-eval/processed/test.subset.jsonl"
  # "<REPO_ROOT>/datasets/compas/processed/test.subset.jsonl"
  # "<REPO_ROOT>/datasets/bbq/processed/test/test.trial.jsonl"
)
names=(
  # "bbq"
  # "crowdspairs"
  # "genMO"
  # "stereoset"
  # "winoqueer"

  "biobias"
  # "discrim-eval"
  # "compas"
  
  # "bbq.random_100"

  # "bbq.trial"
)

# sanity check: equal lengths
if [[ ${#model_paths[@]} -ne ${#model_tags[@]} ]]; then
  echo "Error: model_paths and model_tags must have the same length." >&2; exit 1
fi
if [[ ${#data_paths[@]} -ne ${#names[@]} ]]; then
  echo "Error: data_paths and names must have the same length." >&2; exit 1
fi


# base output dir
base_out="<REPO_ROOT>/results/original.unbiased_messages"

# mode=plain
# mode=shift_reason
export WANDB_MODE=offline

for mode in plain shift_reason
do
for m in "${!model_paths[@]}"; do
  model_path="${model_paths[$m]}"
  model_tag="${model_tags[$m]}"

  echo ">>> MODEL: $model_tag"
  outdir_base="${base_out}/${model_tag}"
  mkdir -p "$outdir_base"

  for i in "${!data_paths[@]}"; do
    data_path="${data_paths[$i]}"
    name="${names[$i]}"
    output_name="${outdir_base}/${name}.original.${model_tag}.${mode}.jsonl"
    mkdir -p "$(dirname "$output_name")"

    echo "=== Running ${name} with ${model_tag} ==="
    echo "data_path:   $data_path"
    echo "output_name: $output_name"
    echo "model_path:  $model_path"
    echo "mode: $mode"

    CUDA_VISIBLE_DEVICES=1 python -m src.inference.inference \
      --model_path "$model_path" \
      --output_name "$output_name" \
      --data_path "$data_path" \
      --gpus_per_worker 1 \
      --num_workers 1 \
      --batch_size 4 \
      --use_fp16 \
      --task_shard_size 16 \
      --no-enable_thinking \
      --message_type unbiased_messages_decision_making \
      --mode "$mode"
  done
done
done
