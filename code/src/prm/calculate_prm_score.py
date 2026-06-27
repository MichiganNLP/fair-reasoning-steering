import argparse
import json
from tqdm import tqdm

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="gsm8k_predictions_formatted.jsonl", help="Output file name")
    parser.add_argument("--injection_phrase", type=str, default=None, help="injection phrase")
    args = parser.parse_args()
    
    avg_score_before_injection = 0
    avg_score_after_injection = 0
    tt_num = 0
    total_lines = sum(1 for _ in open(args.output_path))
    with open(args.output_path, "r") as fin:
        for idx, line in enumerate(tqdm(fin, total=total_lines, desc="Scoring steps")):
            line = json.loads(line)
            
            steps = list(line["step_scores_correspondence"].keys())
            step_scores = list(line["step_scores_correspondence"].values())
            if args.injection_phrase:
                if not any(args.injection_phrase in step for step in steps):
                    # the case where there is no injection (already exceeding the maximum tokens)
                    continue
                tt_num += 1
                # find the injection step index (first match)
                inj_idx = next(
                    i for i, step in enumerate(steps)
                    if args.injection_phrase in step
                )
                line["injection_step_index"] = inj_idx

                # slices exclude the injection step itself
                before_scores = step_scores[:inj_idx]
                after_scores  = step_scores[inj_idx + 1:-1]
                
                if len(after_scores) == 0:
                    continue

                avg_score_before_injection += sum(before_scores) / len(before_scores)
                avg_score_after_injection += sum(after_scores) / len(after_scores)

    print(f"Average score before injection: {avg_score_before_injection * 100/tt_num: .2f}")
    print(f"Average score after injection: {avg_score_after_injection * 100/tt_num: .2f}")
    
