

bbq_filter_entail_path="<REPO_ROOT>/results/bbq/unbiased_messages/bbq.qwen2.5-72b-instruct.inject.filter.entail.jsonl"
bbq_filter_reason_valid_path="<REPO_ROOT>/results/bbq/unbiased_messages/bbq.qwen2.5-72b-instruct.inject.filter.reason_valid.jsonl"
bbq_select_path="<REPO_ROOT>/results/bbq/unbiased_messages/bbq.qwen2.5-72b-instruct.inject.select.jsonl"

# bbq_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen3-8b.no_thinking/bbq.original.qwen3-8b.no_thinking.jsonl"
# crowdspairs_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen3-8b.no_thinking/crowdspairs.original.qwen3-8b.no_thinking.jsonl"
# genmo_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen3-8b.no_thinking/genMO.original.qwen3-8b.no_thinking.jsonl"
# stereoset_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen3-8b.no_thinking/stereoset.original.qwen3-8b.no_thinking.jsonl"
# winoqueer_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen3-8b.no_thinking/winoqueer.original.qwen3-8b.no_thinking.jsonl"

################################################
# sft results
# we reuse the calculation script.
################################################

# bbq_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/bbq.original.qwen2.5-7b-instruct.shift_reason.phrase_4.jsonl"
# crowdspairs_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/crowdspairs.original.qwen2.5-7b-instruct.shift_reason.phrase_4.jsonl"
# genmo_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/genMO.original.qwen2.5-7b-instruct.shift_reason.phrase_4.jsonl"
# stereoset_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/stereoset.original.qwen2.5-7b-instruct.shift_reason.phrase_4.jsonl"
# winoqueer_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/winoqueer.original.qwen2.5-7b-instruct.shift_reason.phrase_4.jsonl"


bbq_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/bbq.original.qwen2.5-7b-instruct.shift_reason.phrase-0.found-by-qwen2.5-7b.run-2.jsonl"
crowdspairs_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/crowdspairs.original.qwen2.5-7b-instruct.shift_reason.phrase-0.found-by-qwen2.5-7b.run-2.jsonl"
genmo_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/genMO.original.qwen2.5-7b-instruct.shift_reason.phrase-0.found-by-qwen2.5-7b.run-2.jsonl"
stereoset_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/stereoset.original.qwen2.5-7b-instruct.shift_reason.phrase-0.found-by-qwen2.5-7b.run-2.jsonl"
winoqueer_plain_inference_path="<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/winoqueer.original.qwen2.5-7b-instruct.shift_reason.phrase-0.found-by-qwen2.5-7b.run-2.jsonl"


# If it is by category, subsample this amount, otherwise subsample the whole dataset by this amount
# SAMPLE_BY_CATEGORY=1
SAMPLE_BY_CATEGORY=100

download_bbq_dir_path="../datasets/bbq/raw"
processed_bbq_dir_path="../datasets/bbq/processed/"

download_crowdspairs_dir_path="../datasets/crowdspairs/raw"
processed_crowdspairs_dir_path="../datasets/crowdspairs/processed/"

download_genmo_dir_path="../datasets/genMO/raw"
# processed_genmo_dir_path="../datasets/genMO/processed/debug.multi-turn.base"
processed_genmo_dir_path="../datasets/genMO/processed/"

download_stereoset_dir_path="../datasets/stereoset/raw"
processed_stereoset_dir_path="../datasets/stereoset/processed/"

download_winoqueer_dir_path="../datasets/winoqueer/raw"
processed_winoqueer_dir_path="../datasets/winoqueer/processed/"

download_biobias_dir_path="../datasets/biobias/raw"
processed_biobias_dir_path="../datasets/biobias/processed/"

download_compas_dir_path="../datasets/compas/raw"
processed_compas_dir_path="../datasets/compas/processed/"

download_discrim_eval_dir_path="../datasets/discrim-eval/raw"
processed_discrim_eval_dir_path="../datasets/discrim-eval/processed/"

INTERACTION_NAME = "bias"

openai_keypath = "../miscelleneous/openai_key.txt"

icl_result_dirpath = "<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/"
# icl_result_dirpath = "<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/"

##########################################
# sft experiments
# data from model demonstrations
##########################################

raw_gpt_5_mini_data = "<REPO_ROOT>/results/original.unbiased_messages/gpt-5-mini/bbq.original.gpt-5-mini.random_100.jsonl"
raw_qwen2_5_7b_data = "<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-72b-instruct/bbq.random_100.original.qwen2.5-72b-instruct.jsonl"
raw_qwen2_5_72b_data = "<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-72b-instruct/bbq.random_100.original.qwen2.5-72b-instruct.jsonl"
raw_manual_modify_data = "<REPO_ROOT>/results/original.unbiased_messages/manual-modification/bbq.random_100.original.qwen2.5-72b-instruct.jsonl"
raw_llama3_8b_data = "<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/bbq.random_100.original.llama3.1-8b-instruct.jsonl"

processed_gpt_5_mini_data = "<REPO_ROOT>/results/original.unbiased_messages/gpt-5-mini/lf-sft.bbq.original.gpt-5-mini.random_100.json"
processed_qwen2_5_7b_data = "<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-72b-instruct/lf-sft.bbq.random_100.original.qwen2.5-72b-instruct.json"
processed_qwen2_5_72b_data = "<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-72b-instruct/lf-sft.bbq.random_100.original.qwen2.5-72b-instruct.json"
processed_manual_modify_data = "<REPO_ROOT>/results/original.unbiased_messages/manual-modification/lf-sft.bbq.random_100.original.qwen2.5-72b-instruct.json"
process_llama3_8b_data = "<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/lf-sft.bbq.random_100.original.llama3.1-8b-instruct.jsonl"
