import json
import os
# from src.inference.calculate_score import get_input_length, get_output_length

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
    
    if isbbq:
        
        with open("<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl", 'r') as f:
            original_file = f.readlines()
        
        original_file = [json.loads(line) for line in original_file]
        original_file = {itm["extra_info"]["uuid"]: itm["extra_info"]["context_condition"] for itm in original_file}
        
        tt_num = 0
        for itm in data:
            itm["uuid"] = itm["example_id"]
            itm["pred"] = itm["answer"]
            itm["gold"] = itm["label"]
            
            if original_file[itm["uuid"]] == "ambig":
                if itm["pred"] == itm["gold"]:
                    correct_num += 1
                tt_num += 1
                
            # input_len += get_input_length(input)
            # output_len += get_output_length(output)
        
        print(f"Acc: {correct_num/tt_num:2f}; Total: {tt_num}")
        # print(f"Overall input len: {input_len}/{total} = {input_len / total:.2f}\n")
        # print(f"Overall output len: {output_len}/{total} = {output_len / total:.2f}\n")
    else:
        for itm in data:
            itm["pred"] = itm["answer"]
            itm["gold"] = itm["label"]
            
            if itm["pred"] == itm["gold"]:
                correct_num += 1

            # input_len += get_input_length(input)
            # output_len += get_output_length(output)
        
        print(f"Acc: {correct_num/len(data):2f}; Total: {len(data)}")
        # print(f"Overall input len: {input_len}/{total} = {input_len / total:.2f}\n")
        # print(f"Overall output len: {output_len}/{total} = {output_len / total:.2f}\n")
    
    


if __name__ == "__main__":
    
    # for model in ["llama3.1-8b-instruct"]:
    for model in ["qwen2.5-7b-instruct"]:          
        for dataset in ["bbq", "crowdspairs", "genMO", "stereoset", "winoqueer"]:
            
            print(model, dataset)
            pred_file = f"<REPO_ROOT>/results/original.unbiased_messages/{model}/{dataset}.original.{model}.self_debias_reprompting.enhanced.jsonl"
            print(pred_file)
            
            calc_answer(pred_file, isbbq=(dataset == "bbq"))
            