import json
import re
import numpy as np
import random

random.seed(42)
from collections import defaultdict, Counter

from statsmodels.stats.contingency_tables import mcnemar

BASE_PATH = "<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/biobias.original.llama3.1-8b-instruct.plain.jsonl"
INJ_PATH  = "<REPO_ROOT>/results/original.unbiased_messages/llama3.1-8b-instruct/biobias.original.llama3.1-8b-instruct.shift_reason.jsonl"


def parse_pred_from_reasoning(reasoning: str):
    """
    Parse the model's prediction from the reasoning string.
    Expects a line like: '### Answer: A' or '### Answer: B' etc.
    Returns 'A'/'B'/'C'/'D' or None if not found.
    """
    for line in reasoning.splitlines():
        line = line.strip()
        if line.startswith("### Answer:"):
            after = line.split(":", 1)[1].strip()
            m = re.match(r"([A-D])", after)
            if m:
                return m.group(1)
    return None


def load_results(path):
    """
    Load a BioBias jsonl file into a dict keyed by uuid:
      results[uuid] = {
        "gold": ...,
        "gender": ...,
        "profession": ...,
        "pred": ...,
        "correct": 0/1
      }
    """
    results = {}
    skipped = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            original = item.get("original", {})

            uuid = original.get("uuid") or item.get("uuid")
            if uuid is None:
                skipped += 1
                continue

            gold = original.get("gold") or item.get("gold")
            gender = original.get("gender", "UNKNOWN")
            profession = original.get("profession_text", "UNKNOWN")
            reasoning = item.get("reasoning", "")

            pred = parse_pred_from_reasoning(reasoning)
            if gold is None or pred is None:
                skipped += 1
                continue

            correct = int(pred == gold)
            results[uuid] = {
                "gold": gold,
                "gender": gender,
                "profession": profession,
                "pred": pred,
                "correct": correct,
            }
    print(f"Loaded {len(results)} examples from {path}, skipped {skipped}.")
    return results


def mcnemar_pvalue(paired):
    """
    paired: list of (correct_baseline, correct_injection), each 0/1
    Returns (p_value, n01, n10) from exact McNemar's test.

    n01 = baseline wrong, injection correct
    n10 = baseline correct, injection wrong
    """
    n01 = sum(1 for b, i in paired if (not b) and i)   # baseline wrong, inj correct
    n10 = sum(1 for b, i in paired if b and (not i))   # baseline correct, inj wrong

    if n01 + n10 == 0:
        # no discordant pairs => models identical on this subset
        # convention: p-value = 1.0
        return 1.0, n01, n10

    table = [[0, n01],
             [n10, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue, n01, n10


def compute_gender_tpr(results, uuids):
    """
    results: dict[uuid] -> {"correct": 0/1, "gender": ...}
    uuids: list of uuids to include
    Returns: dict[gender] -> TPR (accuracy within that gender)
    """
    by_gender = defaultdict(lambda: {"correct": 0, "total": 0})
    for u in uuids:
        g = results[u]["gender"]
        c = results[u]["correct"]
        by_gender[g]["correct"] += c
        by_gender[g]["total"] += 1

    tpr_gender = {}
    for g, cnts in by_gender.items():
        if cnts["total"] > 0:
            tpr_gender[g] = cnts["correct"] / cnts["total"]
    return tpr_gender


def compute_gender_gap(results, uuids):
    """
    Returns (gap, tpr_gender_dict)
    where gap = max_g TPR(g) - min_g TPR(g).
    """
    tpr_gender = compute_gender_tpr(results, uuids)
    if len(tpr_gender) < 2:
        return None, tpr_gender
    vals = list(tpr_gender.values())
    gap = max(vals) - min(vals)
    return gap, tpr_gender


def bootstrap_gap_diff(base, inj, uuids, num_boot=2000, seed=0):
    """
    Paired bootstrap for the difference in gender TPR gaps
    between injection and baseline on a given subset of UUIDs.

    base, inj: dict[uuid] -> {"correct": 0/1, "gender": ...}
    uuids: list of uuids to include in this analysis (e.g., all, or a profession subset)

    Returns:
        gap_base_obs   : observed baseline gap
        gap_inj_obs    : observed injection gap
        gap_diff_obs   : observed change (inj - base)
        diff_mean      : mean bootstrap estimate of gap_diff
        ci_low, ci_high: 95% CI of gap_diff
        p_one_sided    : one-sided p-value for "gap decreased" (inj < base)
        p_two_sided    : two-sided p-value for "gap changed"
    """
    random.seed(seed)
    np.random.seed(seed)

    # observed gaps on full data
    gap_base_obs, _ = compute_gender_gap(base, uuids)
    gap_inj_obs,  _ = compute_gender_gap(inj,  uuids)
    gap_diff_obs = gap_inj_obs - gap_base_obs   # negative means gap decreased

    diffs = []
    n = len(uuids)

    for _ in range(num_boot):
        # sample uuids with replacement
        boot_uuids = [uuids[random.randint(0, n - 1)] for _ in range(n)]

        gb, _ = compute_gender_gap(base, boot_uuids)
        gi, _ = compute_gender_gap(inj,  boot_uuids)

        # if some bootstrap sample only contains one gender, skip
        if gb is None or gi is None:
            continue
        diffs.append(gi - gb)

    diffs = np.array(diffs)
    diff_mean = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # two-sided p-value: how often is |diff| at least as large as |observed diff|?
    p_two_sided = np.mean(np.abs(diffs) >= np.abs(gap_diff_obs))

    # one-sided p-value for *improvement* (gap decreased):
    # H0: gap_diff >= 0 (no improvement or worse), H1: gap_diff < 0
    if gap_diff_obs < 0:
        # fraction of bootstrap samples that contradict improvement (>= 0)
        p_one_sided = np.mean(diffs >= 0)
    else:
        # if observed change is >= 0, one-sided p for "decrease" is 1.0
        p_one_sided = 1.0

    return (gap_base_obs, gap_inj_obs, gap_diff_obs,
            diff_mean, ci_low, ci_high, p_one_sided, p_two_sided)


def bootstrap_avg_gap_diff(base, inj, common_uuids, professions, num_boot=2000, seed=0):
    """
    Bootstrap significance test for the *average* gender TPR gap
    across all professions.

    Returns:
        avg_gap_base_obs
        avg_gap_inj_obs
        avg_gap_diff_obs
        diff_mean
        ci_low, ci_high
        p_one_sided
        p_two_sided
    """
    random.seed(seed)
    np.random.seed(seed)

    # Compute observed gaps per profession
    gaps_base_obs = []
    gaps_inj_obs = []
    for prof in professions:
        prof_uuids = [u for u in common_uuids if base[u]["profession"] == prof]
        gap_b, _ = compute_gender_gap(base, prof_uuids)
        gap_i, _ = compute_gender_gap(inj,  prof_uuids)
        gaps_base_obs.append(gap_b)
        gaps_inj_obs.append(gap_i)

    avg_gap_base_obs = np.mean(gaps_base_obs)
    avg_gap_inj_obs  = np.mean(gaps_inj_obs)
    avg_gap_diff_obs = avg_gap_inj_obs - avg_gap_base_obs

    # bootstrap diffs
    diffs = []
    n = len(common_uuids)
    ids = common_uuids

    for _ in range(num_boot):
        boot_uuids = [ids[random.randint(0, n-1)] for _ in range(n)]
        gaps_b = []
        gaps_i = []
        for prof in professions:
            pu = [u for u in boot_uuids if base[u]["profession"] == prof]
            gb, _ = compute_gender_gap(base, pu)
            gi, _ = compute_gender_gap(inj,  pu)
            # skip if any profession only has one gender in sample
            if gb is None or gi is None:
                break
            gaps_b.append(gb)
            gaps_i.append(gi)
        if len(gaps_b) != len(professions):
            continue
        diffs.append(np.mean(gaps_i) - np.mean(gaps_b))

    diffs = np.array(diffs)
    diff_mean = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # p-values
    p_two_sided = np.mean(np.abs(diffs) >= np.abs(avg_gap_diff_obs))
    if avg_gap_diff_obs < 0:
        p_one_sided = np.mean(diffs >= 0)   # probability of no improvement
    else:
        p_one_sided = 1.0

    return (avg_gap_base_obs, avg_gap_inj_obs, avg_gap_diff_obs,
            diff_mean, ci_low, ci_high, p_one_sided, p_two_sided)


def main():
    base = load_results(BASE_PATH)
    inj  = load_results(INJ_PATH)

    common_uuids = sorted(set(base.keys()) & set(inj.keys()))
    print(f"\nCommon UUIDs: {len(common_uuids)}")

    # ================================
    # Overall accuracy + McNemar test
    # ================================
    paired_overall = []
    base_correct_total = 0
    inj_correct_total = 0

    for u in common_uuids:
        cb = base[u]["correct"]
        ci = inj[u]["correct"]
        base_correct_total += cb
        inj_correct_total += ci
        paired_overall.append((cb, ci))

    n = len(common_uuids)
    acc_base = base_correct_total / n
    acc_inj  = inj_correct_total  / n
    p_overall, n01, n10 = mcnemar_pvalue(paired_overall)

    print("\n=== Overall Accuracy ===")
    print(f"Baseline (N/A): {acc_base*100:.2f}%")
    print(f"Injection (INJ): {acc_inj*100:.2f}%")
    print(f"McNemar: n01={n01}, n10={n10}, p={p_overall:.4g}")

    # ======================================
    # Group-level TPR (accuracy) by gender
    # ======================================
    print("\n=== Group-Level TPR (Accuracy) by Gender + McNemar ===")

    # group indices
    gender_to_uuids = defaultdict(list)
    for u in common_uuids:
        g = base[u]["gender"]  # assume same in inj
        gender_to_uuids[g].append(u)

    for g, uuids in gender_to_uuids.items():
        paired = []
        cb_sum = 0
        ci_sum = 0
        for u in uuids:
            cb = base[u]["correct"]
            ci = inj[u]["correct"]
            cb_sum += cb
            ci_sum += ci
            paired.append((cb, ci))
        n_g = len(uuids)
        tpr_base = cb_sum / n_g
        tpr_inj  = ci_sum / n_g
        p_g, n01_g, n10_g = mcnemar_pvalue(paired)

        print(f"\nGender = {g}")
        print(f"  N = {n_g}")
        print(f"  TPR_base: {tpr_base*100:.2f}%")
        print(f"  TPR_inj : {tpr_inj*100:.2f}%")
        print(f"  McNemar: n01={n01_g}, n10={n10_g}, p={p_g:.4g}")

    # ====================================================
    # Profession x Gender TPRs + McNemar (fairness cells)
    # ====================================================
    print("\n=== TPR by Profession × Gender + McNemar ===")
    prof_gender_to_uuids = defaultdict(list)
    for u in common_uuids:
        p = base[u]["profession"]
        g = base[u]["gender"]
        prof_gender_to_uuids[(p, g)].append(u)

    # sort by profession then gender for readability
    for (p, g), uuids in sorted(prof_gender_to_uuids.items()):
        paired = []
        cb_sum = 0
        ci_sum = 0
        for u in uuids:
            cb = base[u]["correct"]
            ci = inj[u]["correct"]
            cb_sum += cb
            ci_sum += ci
            paired.append((cb, ci))
        n_pg = len(uuids)
        if n_pg == 0:
            continue
        tpr_base = cb_sum / n_pg
        tpr_inj  = ci_sum / n_pg
        p_pg, n01_pg, n10_pg = mcnemar_pvalue(paired)

        print(f"\nProfession = {p}, Gender = {g}")
        print(f"  N = {n_pg}")
        print(f"  TPR_base: {tpr_base*100:.2f}%")
        print(f"  TPR_inj : {tpr_inj*100:.2f}%")
        print(f"  McNemar: n01={n01_pg}, n10={n10_pg}, p={p_pg:.4g}")
        
    print("\n=== Bootstrap Test for Global Gender TPR Gap Difference ===")
    (gap_base_obs, gap_inj_obs, gap_diff_obs,
    diff_mean, ci_low, ci_high, p_one_sided, p_two_sided) = bootstrap_gap_diff(
        base, inj, common_uuids, num_boot=2000, seed=0
    )

    print(f"Observed gender TPR gap (baseline):  {gap_base_obs*100:.2f} pp")
    print(f"Observed gender TPR gap (injection): {gap_inj_obs*100:.2f} pp")
    print(f"Observed change (inj - base):        {gap_diff_obs*100:.2f} pp")
    print(f"Bootstrap mean change:               {diff_mean*100:.2f} pp")
    print(f"95% CI for change:                   [{ci_low*100:.2f}, {ci_high*100:.2f}] pp")
    print(f"One-sided p-value (gap decreased):   {p_one_sided:.4f}")
    print(f"Two-sided p-value (gap changed):     {p_two_sided:.4f}")


    
    # -----------------------------
    # Bootstrap Test for TPR Gap Difference per Profession
    # -----------------------------
    print("\n=== Bootstrap Test of Gender TPR Gap per Profession ===")

    professions = ["nurse", "physician", "professor", "teacher"]

    for prof in professions:
        prof_uuids = [u for u in common_uuids if base[u]["profession"] == prof]

        if len(prof_uuids) < 10:
            print(f"\nProfession = {prof}: too few examples ({len(prof_uuids)}) to test.")
            continue

        (gap_base_obs, gap_inj_obs, gap_diff_obs,
        diff_mean, ci_low, ci_high, p_one_sided, p_two_sided) = bootstrap_gap_diff(
            base, inj, prof_uuids, num_boot=2000, seed=0
        )

        print(f"\nProfession = {prof}")
        print(f"  N = {len(prof_uuids)}")
        print(f"  Baseline gender TPR gap:   {gap_base_obs*100:.2f} pp")
        print(f"  Injection gender TPR gap:  {gap_inj_obs*100:.2f} pp")
        print(f"  Observed change (inj-base):{gap_diff_obs*100:.2f} pp")
        print(f"  Bootstrap mean change:     {diff_mean*100:.2f} pp")
        print(f"  95% CI:                    [{ci_low*100:.2f}, {ci_high*100:.2f}] pp")
        print(f"  One-sided p (gap decrease):{p_one_sided:.4f}")
        print(f"  Two-sided p (gap change):  {p_two_sided:.4f}")


    print("\n=== Bootstrap Significance Test for Average TPR Gap Across Professions ===")

    professions = ["nurse", "physician", "professor", "teacher"]

    (avg_base, avg_inj, avg_diff,
    diff_mean, ci_low, ci_high,
    p_one_sided, p_two_sided) = bootstrap_avg_gap_diff(
        base, inj, common_uuids, professions
    )

    print(f"Avg gap (baseline):  {avg_base*100:.2f} pp")
    print(f"Avg gap (injection): {avg_inj*100:.2f} pp")
    print(f"Observed change:     {avg_diff*100:.2f} pp")
    print(f"Bootstrap mean diff: {diff_mean*100:.2f} pp")
    print(f"95% CI:              [{ci_low*100:.2f}, {ci_high*100:.2f}] pp")
    print(f"One-sided p:         {p_one_sided:.4f}")
    print(f"Two-sided p:         {p_two_sided:.4f}")


    # You can optionally post-process the TPRs per profession to recompute
    # ΔTPR per profession and see if the direction of changes is consistent
    # with the aggregate gaps you report in the table.

if __name__ == "__main__":
    main()
