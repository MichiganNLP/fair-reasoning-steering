import json
import os
from src.inference.calculate_score import get_input_length, get_output_length

def calc_answer(filepath, isbbq = False):
    with open(filepath, 'r') as f:
        raw_data = f.readlines()
    
    data = []
    for line in raw_data:
        try:
            data.append(json.loads(line))
        except Exception:
            line = line.strip("\"")
            data.append(json.loads(line))

    correct_num = 0
    input_len = 0
    output_len = 0
    total = len(data)
    num_rounds = 0
    
    if isbbq:
        with open("<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl", 'r') as f:
            original_file = f.readlines()
        
        original_file = [json.loads(line) for line in original_file]
        original_file = {itm["extra_info"]["uuid"]: itm["extra_info"]["context_condition"] for itm in original_file}
        
        tt_num = 0
        for itm in data:
            for log in itm["logs"]:
                input = log.split("assistant\n")[0]
                output = log.split("assistant\n")[-1]
                input_len += get_input_length(input)
                output_len += get_output_length(output)
            num_rounds += len(itm["logs"])
            if itm["new_log"]:
                input = itm["new_log"].split("assistant\n")[0]
                output = itm["new_log"].split("assistant\n")[-1]
                input_len += get_input_length(input)
                output_len += get_output_length(output)
                num_rounds += 1
                
            if original_file[itm["example_id"]] == "ambig":
                if itm["answer"] == itm["label"]:
                    correct_num += 1
                tt_num += 1
        
        print(f"Acc: {correct_num/tt_num:2f}; Total: {tt_num}")
        print(f"Overall input len: {input_len}/{total} = {input_len / total:.2f}\n")
        print(f"Overall output len: {output_len}/{total} = {output_len / total:.2f}\n")
        print(f"Overall round number: {num_rounds}/{total} = {num_rounds / total:.2f}\n")
    else:
        for itm in data:
            for log in itm["logs"]:
                input = log.split("assistant\n")[0]
                output = log.split("assistant\n")[-1]
                input_len += get_input_length(input)
                output_len += get_output_length(output)
                num_rounds += len(itm["logs"])
                
            if itm["new_log"]:
                input = itm["new_log"].split("assistant\n")[0]
                output = itm["new_log"].split("assistant\n")[-1]
                input_len += get_input_length(input)
                output_len += get_output_length(output)
                num_rounds += 1
                
            if itm["answer"] == itm["label"]:
                correct_num += 1
        
        print(f"Acc: {correct_num/len(data):2f}; Total: {len(data)}")
        print(f"Overall input len: {input_len}/{total} = {input_len / total:.2f}\n")
        print(f"Overall output len: {output_len}/{total} = {output_len / total:.2f}\n")
        print(f"Overall round number: {num_rounds}/{total} = {num_rounds / total:.2f}\n")


if __name__ == "__main__":
    
    for model in ["llama3.1-8b-instruct"]:
                #   , "qwen2.5-7b-instruct"]:
        for dataset in ["bbq", "crowdspairs", "genMO", "stereoset", "winoqueer"]:
            
            print(model, dataset)
            pred_file = f"<REPO_ROOT>/results/original.unbiased_messages/{model}/{dataset}.original.{model}.adbp.jsonl"
            print(pred_file)
            
            calc_answer(pred_file, isbbq=(dataset == "bbq"))
            