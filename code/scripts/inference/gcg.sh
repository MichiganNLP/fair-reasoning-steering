
# model_path=<HF_HOME>/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
# model_path=<HF_HOME>/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
model_path="<HF_HOME>/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
test_path=<REPO_ROOT>/datasets/bbq/processed/test/test.100.random.jsonl
# test_path=<REPO_ROOT>/datasets/bbq/processed/test/test.20.random.jsonl

train_dir=<REPO_ROOT>/datasets/bbq/processed/train

# All train sizes
# sizes=(10 20 50 100 200 500 1000)
sizes=(100)

declare -a seeds=(
    # '\n\nWait, am I being fair?'

  # Fairness-explicit
#   $'\n\nWait, am I being fair?'
  $'\n\nCould this decision be unfair or biased?'
  $'\n\nLet me check that I am treating all options equally.'
  $'\n\nI should avoid stereotypes and focus only on the evidence.'

# #   # Meta-reasoning
  $'\n\nWait, let me double-check my reasoning step by step.'
  $'\n\nI should carefully reconsider each answer choice.'
  $'\n\nLet me verify whether my previous reasoning still holds.'
  $'\n\nI should slow down and think through the options again.'

#   # Generic carefulness / boilerplate
  $'\n\nPlease answer carefully, logically, and consistently.'
  $'\n\nI should give a careful, well-justified answer.'
  $'\n\nI must follow the instructions and avoid careless mistakes.'

#   # Noisy / weird
  $'\n\nWHAT???? SHOULDAT being fair?诸如'
  $'\n\nыџNWaitemsp am\tnot being fair?'
  $'\n\nWait MEQUI_none being fair?(Parameter teammates'
  $'\n\nWaitnowrap Since\tnot being fair revisit…………'
  $'\n\n### ??? !!! %% $$ @@'

#   # Near-null
  $'\n\n'
  $'\n\n   '
)

# declare -a lens=(4 8 10 16 32)
declare -a lens=(4 8 10)
# inj_len="${INJ_LEN:-10}"
# inj_len=10
echo "Using inj_len=${inj_len}"

for inj_len in "${lens[@]}"; do
for seed_text in "${seeds[@]}"; do
    for size in "${sizes[@]}"; do
        train_path="$train_dir/train.${size}.random.jsonl"
        log_path="<REPO_ROOT>/gcg_logs/true-objective-$inj_len.version-1.top50.log"

        # for run in 1 2 3; do
        echo "========================================"
        echo "Running GCG"
        echo "Train file: $train_path"
        echo "Train size: $size"
        echo "Running inj_len=${inj_len}, seed_text='${seed_text}'"
        # echo "Run number: $run"
        echo "========================================"

        CUDA_VISIBLE_DEVICES=0,1 python -m src.inference.gcg \
            --model_path "$model_path" \
            --train_path "$train_path" \
            --test_path "$test_path" \
            --passes 4 \
            --seed_text "$seed_text" \
            --inj_len "$inj_len" \
            --log_path "$log_path"
        # done
    done
done
done