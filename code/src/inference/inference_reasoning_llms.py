#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_jsonl_or_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    # allow either JSONL or a single JSON list/dict
    if "\n" in txt and txt.lstrip()[0] != "[":
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    obj = json.loads(txt)
    if isinstance(obj, list):
        return obj
    return [obj]


@torch.inference_mode()
def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # only decode the newly generated part (after the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def shift_reason(
    model_path: str,
    data_path: str,
    output_name: str,
    injection_phrase: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: str,
    dtype: str,
):
    # --- load model/tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # ensure pad token exists (common for some LLaMA-like tokenizers)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "auto" else None,
    )

    if device != "auto":
        model = model.to(device)

    model.eval()

    # --- load data ---
    data = load_jsonl_or_json(data_path)

    os.makedirs(os.path.dirname(output_name) or ".", exist_ok=True)
    with open(output_name, "w", encoding="utf-8") as out_f:
        for itm in data:
            prompt = itm.get("prompt", "")
            reasoning = itm.get("reasoning", "")

            # keep everything up to (but excluding) the final </think> chunk
            if "</think>" in reasoning:
                parts = reasoning.split("</think>")
                truncated_reasoning = "</think>".join(parts[:-1]) if len(parts) > 1 else reasoning

            elif "assistantfinal" in reasoning:
                parts = reasoning.split("assistantfinal")
                truncated_reasoning = "assistantfinal".join(parts[:-1]) if len(parts) > 1 else reasoning
            else:
                
                if "gpt-oss" in model_path:
                    reasoning = ".".join(reasoning.split(".")[:-2])

            
            new_prompt = prompt + truncated_reasoning + "\n\n" + injection_phrase

            completion = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=new_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            
            old_reasoning = itm.pop("reasoning")
            
            record = {
                **itm,
                "old_reasoning": old_reasoning,
                "new_prompt": new_prompt,
                "reasoning": completion,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF id for the model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="JSON or JSONL containing prompt/reasoning fields")
    parser.add_argument("--output_name", type=str, required=True, help="Output JSONL path")

    # NOTE: this should be the literal phrase, not a path
    parser.add_argument("--injection_phrase", type=str, required=True, help="Text to append after truncated reasoning")

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling; default is greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu (auto uses device_map=auto)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    args = parser.parse_args()

    shift_reason(
        model_path=args.model_path,
        data_path=args.data_path,
        output_name=args.output_name,
        injection_phrase=args.injection_phrase,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=args.device,
        dtype=args.dtype,
    )
