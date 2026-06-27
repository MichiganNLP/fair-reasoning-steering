files=(
<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/bbq.original.llama3.1-8b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/bbq.original.llama3.1-8b-instruct.shift_reason.phrase-1.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/bbq.original.qwen2.5-7b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/bbq.original.qwen2.5-7b-instruct.shift_reason.phrase-1.jsonl

<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/crowdspairs.original.llama3.1-8b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/crowdspairs.original.llama3.1-8b-instruct.shift_reason.phrase-1.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/crowdspairs.original.qwen2.5-7b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/crowdspairs.original.qwen2.5-7b-instruct.shift_reason.phrase-1.jsonl

<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/genMO.original.llama3.1-8b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/genMO.original.llama3.1-8b-instruct.shift_reason.phrase-1.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/genMO.original.qwen2.5-7b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/genMO.original.qwen2.5-7b-instruct.shift_reason.phrase-1.jsonl

<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/stereoset.original.llama3.1-8b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/stereoset.original.llama3.1-8b-instruct.shift_reason.phrase-1.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/stereoset.original.qwen2.5-7b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/stereoset.original.qwen2.5-7b-instruct.shift_reason.phrase-1.jsonl

<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/winoqueer.original.llama3.1-8b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/winoqueer.original.llama3.1-8b-instruct.shift_reason.phrase-1.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/winoqueer.original.qwen2.5-7b-instruct.shift_reason.jsonl
<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/winoqueer.original.qwen2.5-7b-instruct.shift_reason.phrase-1.jsonl
)

injection_phrase=(
"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"
"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"

"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"
"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"

"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"
"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"

"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"
"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"

"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"
"Wait, am I being fair?"
"Wait quienes am I being subjective; striped"
)

for i in "${!files[@]}"; do
    file="${files[$i]}"
    phrase="${injection_phrase[$i]}"

    output_file="${file%.jsonl}.prm-rate.jsonl"

    echo "Processing $file"
    echo "Injection phrase: $phrase"
    echo "Output: $output_file"

    python -m src.prm.rate_reasoning_chain \
        --file_path "$file" \
        --output_path "$output_file" \
        --injection_phrase "$phrase"

done
