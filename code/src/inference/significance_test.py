import json
import os
from typing import Dict, Tuple, List
import math
import argparse

########################################
# Your original helpers (slightly trimmed)
########################################

def extract_answer(text: str):
    if "### Reasoning" in text:
        text = text.split("### Reasoning")[-1]
    if "### Answer:" in text:
        if "### Answer: A" in text:
            return "A"
        elif "### Answer: B" in text:
            return "B"
        elif "### Answer: C" in text:
            return "C"
    return None


def g(d: dict, key: str, default=None):
    """Get key from dict; if missing, try d['extra_info'][key]; else default."""
    if key in d:
        return d[key]
    else:
        try_extra_info = d.get("extra_info", {})
        if try_extra_info:
            return try_extra_info.get(key)
        return default


########################################
# McNemar's test (no SciPy needed)
########################################

def mcnemar_from_correctness(
    correct_a: List[bool],
    correct_b: List[bool],
    continuity: bool = True,
) -> Tuple[int, int, int, int, float, float]:
    """
    Run McNemar's test given two boolean correctness arrays.

    correct_a[i] = True if model A is correct on example i
    correct_b[i] = True if model B is correct on example i

    Returns:
        a, b, c, d, chi2_stat, p_value
        where:
          a = both correct
          b = A correct, B wrong
          c = A wrong,  B correct
          d = both wrong
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length.")

    a = b = c = d = 0
    for ca, cb in zip(correct_a, correct_b):
        if ca and cb:
            a += 1
        elif ca and (not cb):
            b += 1
        elif (not ca) and cb:
            c += 1
        else:
            d += 1

    if b + c == 0:
        # No discordant pairs => identical behavior
        chi2_stat = 0.0
        p_value = 1.0
        return a, b, c, d, chi2_stat, p_value

    if continuity:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2_stat = (b - c) ** 2 / (b + c)

    # For df=1, p-value = erfc(sqrt(chi2)/sqrt(2))
    z = math.sqrt(chi2_stat)
    p_value = math.erfc(z / math.sqrt(2.0))
    return a, b, c, d, chi2_stat, p_value


########################################
# Loading correctness from your JSONL format
########################################

def load_correctness_by_key(jsonl_path: str, dataset_name: str) -> Dict[Tuple, bool]:
    """
    Load a JSONL file in your format and return:
        key -> correctness (True/False)
    where key = (ex_id, q_id, context_condition)

    Uses your dedup key logic and extract_answer.
    """
    letter_to_index = {"A": 0, "B": 1, "C": 2}
    idx_to_letter = {v: k for k, v in letter_to_index.items()}

    seen_keys = set()
    correctness_by_key: Dict[Tuple, bool] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):                
            ex = json.loads(line)
            if dataset_name.lower() == "bbq":
                if "original" in ex and ex["original"]["extra_info"]["context_condition"] != "ambig":
                    continue
            original = ex.get("original")
            reasoning = None
            if "reasoning" in ex: 
                reasoning = ex["reasoning"]
            elif "reasoning_second" in ex:
                reasoning = ex["reasoning_second"]

            # Dedup key (same as in your original script)
            context_condition = None
            if original:
                context_condition = original.get("extra_info")
            
            if context_condition:
                context_condition = context_condition.get("context_condition")

            dedup_key = None
            if original and "extra_info" in original:
                dedup_key = (original["extra_info"]["uuid"], None, context_condition)
            elif "uuid" in ex:
                if dataset_name.lower() == "bbq":
                    dedup_key = (ex["uuid"], None, 'ambig')
                else:
                    dedup_key = (ex["uuid"], None, '')
            elif "example_id" in ex:
                if dataset_name.lower() == "bbq":
                    dedup_key = (ex["example_id"], None, 'ambig')
                else:
                    dedup_key = (ex["example_id"], None, '')
            
            if dedup_key in seen_keys:
                # skip duplicates
                continue
            seen_keys.add(dedup_key)

            # Gold label: letter A/B/C
            gold_letter = None  
            if original:
                gold_letter = original.get("groundtruth") if "groundtruth" in original else original.get("gold")
            elif "gold" in ex:
                gold_letter = ex["gold"]
            elif "label" in ex:
                gold_letter = ex["label"]
            gold_idx = letter_to_index.get(gold_letter, None)

            if gold_idx is None:
                # no valid gold label, skip
                continue

            # Parse prediction
            if reasoning:
                pred_letter = extract_answer(reasoning)
            elif "pred" in ex:
                pred_letter = ex["pred"]
            elif "answer" in ex:
                pred_letter = ex["answer"]

            # correctness
            is_correct = (pred_letter == gold_letter)
            correctness_by_key[dedup_key] = is_correct

    return correctness_by_key


########################################
# Main: compare two models for each dataset
########################################

if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Fill these with your two sets of JSONL paths (Model A vs Model B)
    # They should be in the SAME order across the two lists.
    # For example:
    #   model_a_paths = [bbq_plain_inference_path_modelA, ...]
    #   model_b_paths = [bbq_plain_inference_path_modelB, ...]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_path", type=str)
    parser.add_argument("--model_b_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    args = parser.parse_args()
    
    jsonl_path_a = args.model_a_path
    jsonl_path_b = args.model_b_path
    dataset_name = args.dataset_name
    
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Model A file: {jsonl_path_a}")
    print(f"Model B file: {jsonl_path_b}")

    corr_a_by_key = load_correctness_by_key(jsonl_path_a, dataset_name)
    corr_b_by_key = load_correctness_by_key(jsonl_path_b, dataset_name)

    # Intersect keys to ensure pairing
    common_keys = sorted(set(corr_a_by_key.keys()) & set(corr_b_by_key.keys()))
    # breakpoint()
    if not common_keys:
        print("No overlapping examples between the two models. Skipping.")
        exit(0)

    correct_a = [corr_a_by_key[k] for k in common_keys]
    correct_b = [corr_b_by_key[k] for k in common_keys]

    n = len(common_keys)
    acc_a = sum(correct_a) / n
    acc_b = sum(correct_b) / n

    a, b, c, d, chi2_stat, p_value = mcnemar_from_correctness(correct_a, correct_b, continuity=True)

    print(f"Total paired examples: {n}")
    print(f"Model A accuracy: {acc_a * 100:.2f}%")
    print(f"Model B accuracy: {acc_b * 100:.2f}%")
    print()
    print("Contingency table (per-example correctness):")
    print(f"  a = both correct          : {a}")
    print(f"  b = A correct, B wrong    : {b}")
    print(f"  c = A wrong,  B correct   : {c}")
    print(f"  d = both wrong            : {d}")
    print()
    print(f"McNemar chi-square statistic: {chi2_stat:.4f}")
    print(f"p-value                     : {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"Result: DIFFERENCE IS STATISTICALLY SIGNIFICANT (p < {alpha}).")
    else:
        print(f"Result: difference is NOT statistically significant (p >= {alpha}).")
