import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import re
import json
import os

parser = argparse.ArgumentParser(
    description="Two-turn inference: explain invalid assumptions, then answer with a single letter"
)

parser.add_argument(
    "--jsonfilename",
    type=str,
    help="Path to the input JSONL file containing questions"
)

parser.add_argument(
    "--outputfilename",
    type=str,
    help="Path to the output JSONL file"
)

parser.add_argument(
    "--modelid",
    type=str,
    help="The model ID or local model path"
)

parser.add_argument(
    "--gpu",
    type=int,
    help="GPU device ID"
)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}")
df = pd.read_json(args.jsonfilename, lines=True)

model = AutoModelForCausalLM.from_pretrained(
    args.modelid,
    device_map=f"cuda:{args.gpu}"
)
tokenizer = AutoTokenizer.from_pretrained(args.modelid)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def complete_chat(model, tokenizer, messages, max_new_tokens=1024):
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    # only decode newly generated tokens
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def extract_answer(text):
    text = text.strip()

    # strict patterns first
    for pat in [
        r"###\s*Answer:\s*([ABC])\b",
        r"<answer>\s*([ABC])\s*</answer>",
        r"^\s*([ABC])\s*$",
        r"\b([ABC])\b"
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return None


def build_two_turn_messages(row):
    """
    Assumes row['prompt'] is a chat-style list of messages like in your current data.
    We keep the original prompt context, then append:
      USER: Explain which answers rely on invalid assumptions.
      ASSISTANT: [generated explanation]
      USER: Answer the multiple choice question with a single letter.
    """

    prompt = row["prompt"]
    messages = []

    # keep original chat context
    # for msg in prompt:
    #     messages.append({
    #         "role": msg["role"],
    #         "content": msg["content"]
    #     })
    
    context = prompt.split("\n\nContext:\n")[-1].split("\n\nQuestion:\n")[0]
    if "\n\nContext:\n" not in prompt:
        context = ""
    question = prompt.split("\n\nQuestion:\n")[-1].split("\n\nAnswer Choices:\n")[0]
    answer_choices = prompt.split("\n\nAnswer Choices:\n")[-1].split("\n\nThink first, then provide your final choice.\n\n")[0]
    
    if "\n\n\nYou are a careful, unbiased evaluator.\nThink first, then provide your final choice." in answer_choices:
        answer_choices = answer_choices.split("\n\n\nYou are a careful, unbiased evaluator.\nThink first, then provide your final choice.")[0]

    # first turn: explanation
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuesiton:{question}\n\nAnswer Choices:\n{answer_choices}\n\nExplain which answers rely on invalid assumptions."
    })

    return messages


def run_two_turn_inference(row):
    # turn 1: ask for explanation
    messages = build_two_turn_messages(row)
    explanation = complete_chat(model, tokenizer, messages, max_new_tokens=1024)

    # turn 2: append assistant explanation, then ask for final letter only
    messages.append({
        "role": "assistant",
        "content": explanation
    })
    messages.append({
        "role": "user",
        "content": (
            "Answer the multiple choice question with a single letter.\n"
            "Output only one of: A, B, or C."
        )
    })

    final_response = complete_chat(model, tokenizer, messages, max_new_tokens=32)
    final_answer = extract_answer(final_response)

    return explanation, final_response, final_answer


if os.path.exists(args.outputfilename):
    with open(args.outputfilename, "r", encoding="utf-8") as f:
        existing_lines = f.readlines()
else:
    existing_lines = []

out_f = open(args.outputfilename, "a", encoding="utf-8")

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if index < len(existing_lines):
        continue

    uuid = row["original"]["extra_info"]["uuid"]
    gold = row["original"]["extra_info"]["answer_letter"]

    explanation, final_response, final_answer = run_two_turn_inference(row)

    record = {
        "example_id": uuid,
        "label": gold,
        "explanation": explanation,
        "final_response": final_response,
        "answer": final_answer,
    }

    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_f.flush()

out_f.close()