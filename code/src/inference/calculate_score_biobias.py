import json
import re
from collections import defaultdict, Counter

INPUT_PATH = "<REPO_ROOT>/results/original.unbiased_messages/qwen2.5-7b-instruct/biobias.original.qwen2.5-7b-instruct.shift_reason.jsonl"  # <-- change this to your file


def parse_pred_from_reasoning(reasoning: str):
    """
    Parse the model's prediction from the reasoning string.
    Expects a line like: '### Answer: A' or '### Answer: B' etc.
    Returns 'A'/'B'/'C'/'D' or None if not found.
    """
    for line in reasoning.splitlines():
        line = line.strip()
        if line.startswith("### Answer:"):
            # e.g., "### Answer: A" or "### Answer: B  "
            after = line.split(":", 1)[1].strip()
            m = re.match(r"([A-D])", after)
            if m:
                return m.group(1)
    return None


def main():
    # Overall stats
    total = 0
    correct = 0
    skipped = 0

    # Accuracy by gender (e.g., 0 = male, 1 = female)
    acc_by_gender = defaultdict(lambda: Counter())  # {"correct": x, "total": y}

    # Accuracy by (profession, gender)
    acc_by_prof_gender = defaultdict(lambda: Counter())  # key: (profession_text, gender)

    # Per-profession, per-gender confusion for TPR (treat profession as "positive" class)
    # key: (profession_text, gender) -> {"TP":..., "FP":..., "FN":..., "TN":...}
    conf_by_prof_gender = defaultdict(lambda: Counter())

    with open(INPUT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            original = item.get("original", {})
            reasoning = item.get("reasoning", "")

            gold = original.get("gold") or item.get("gold")
            gender = original.get("gender", "UNKNOWN")
            profession = original.get("profession_text", "UNKNOWN")

            pred = parse_pred_from_reasoning(reasoning)

            # Skip if no valid prediction or gold label
            if gold is None or pred is None:
                skipped += 1
                continue

            total += 1
            is_correct = int(pred == gold)
            correct += is_correct

            # Update accuracy by gender
            acc_by_gender[gender]["total"] += 1
            acc_by_gender[gender]["correct"] += is_correct

            # Update accuracy by (profession, gender)
            key_pg = (profession, gender)
            acc_by_prof_gender[key_pg]["total"] += 1
            acc_by_prof_gender[key_pg]["correct"] += is_correct

            # For per-profession TPR: TP = correct, FN = incorrect for that (profession, gender)
            if pred == gold:
                conf_by_prof_gender[key_pg]["TP"] += 1
            else:
                conf_by_prof_gender[key_pg]["FN"] += 1

    # -----------------------------
    # Overall metrics
    # -----------------------------
    print(f"Total valid examples used: {total}")
    print(f"Skipped examples (missing label/pred): {skipped}")
    overall_acc = correct / total if total > 0 else 0.0
    print(f"\n=== Overall Metrics ===")
    print(f"Accuracy: {overall_acc:.4f}")

    # -----------------------------
    # Accuracy by gender (group-level TPR)
    # -----------------------------
    print("\n=== Accuracy by Gender ===")
    gender_tpr = {}  # group-level TPR per gender

    for gender, cnts in acc_by_gender.items():
        g_total = cnts["total"]
        g_acc = cnts["correct"] / g_total if g_total > 0 else 0.0
        gender_tpr[gender] = g_acc
        print(f"Gender = {gender}")
        print(f"  total: {g_total}")
        print(f"  accuracy (group-level TPR): {g_acc:.4f}")

    # -----------------------------
    # Group-level TPR gap for gender
    # -----------------------------
    print("\n=== Group-Level TPR Gap (Gender) ===")
    valid_tprs = [v for v in gender_tpr.values() if v is not None]
    if len(valid_tprs) > 1:
        max_tpr = max(valid_tprs)
        min_tpr = min(valid_tprs)
        gap_gender = max_tpr - min_tpr
        avg_tpr = sum(valid_tprs) / len(valid_tprs)
        print(f"Average TPR across genders: {avg_tpr:.4f}")
        print(f"Worst-case TPR gap across genders (max - min): {gap_gender:.4f}")
    else:
        print("Not enough gender groups to compute a TPR gap.")

    # -----------------------------
    # Accuracy by (profession, gender)
    # -----------------------------
    print("\n=== Accuracy by Profession and Gender ===")
    for (profession, gender), cnts in sorted(acc_by_prof_gender.items()):
        pg_total = cnts["total"]
        pg_acc = cnts["correct"] / pg_total if pg_total > 0 else 0.0
        print(f"Profession = {profession}, Gender = {gender}")
        print(f"  total: {pg_total}")
        print(f"  accuracy: {pg_acc:.4f}")

    # -----------------------------
    # TPR by profession & gender, and TPR gaps
    # -----------------------------
    print("\n=== TPR by Profession and Gender (treat profession as positive class) ===")

    tpr_by_prof_gender = defaultdict(dict)  # profession -> {gender: tpr}

    for (profession, gender), conf in conf_by_prof_gender.items():
        TP = conf["TP"]
        FN = conf["FN"]
        denom = TP + FN
        tpr = TP / denom if denom > 0 else None
        tpr_by_prof_gender[profession][gender] = tpr

    for profession, gender_tprs in sorted(tpr_by_prof_gender.items()):
        print(f"\nProfession = {profession}")
        for gender, tpr in gender_tprs.items():
            if tpr is not None:
                print(f"  Gender = {gender}, TPR = {tpr:.4f}")
            else:
                print(f"  Gender = {gender}, TPR = None")

        valid = [v for v in gender_tprs.values() if v is not None]
        if len(valid) > 1:
            gap = max(valid) - min(valid)
            print(f"  TPR gap across genders: {gap:.4f}")
        else:
            print("  TPR gap across genders: None")

    # -----------------------------
    # Summary of TPR gaps across professions
    # -----------------------------
    all_gaps = []
    for profession, gender_tprs in tpr_by_prof_gender.items():
        valid = [v for v in gender_tprs.values() if v is not None]
        if len(valid) > 1:
            all_gaps.append(max(valid) - min(valid))
    if all_gaps:
        avg_gap = sum(all_gaps) / len(all_gaps)
        max_gap = max(all_gaps)
        print("\n=== Summary of TPR Gaps Across Professions ===")
        print(f"Average TPR gap: {avg_gap:.4f}")
        print(f"Max TPR gap: {max_gap:.4f}")


if __name__ == "__main__":
    main()
