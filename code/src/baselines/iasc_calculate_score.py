import json
import os
from src.inference.calculate_score import get_input_length, get_output_length, tokenizer
from src.other_methods.iasc import TURN2_FEEDBACK_TEMPLATE, TURN3_REFINE_TEMPLATE, build_chat_messages


def calculate_input_tokens(messages):
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True, )
    num_tokens = inputs["input_ids"].shape[-1]
    return num_tokens

def turn_2_input_len(initial_reasoning, question_text):
    resp1 = initial_reasoning

    messages = build_chat_messages(
        "You will be shown a question and a response to it. "
        "You are an evaluator."
    )
    messages.append({"role": "assistant", "content": resp1})

    # -------------------------
    # Turn 2: feedback
    # -------------------------
    turn2_user = TURN2_FEEDBACK_TEMPLATE.format(question=question_text, response=resp1)
    messages.append({"role": "user", "content": turn2_user})
    
    return calculate_input_tokens(messages), messages


def turn_3_input_len(question_text, resp1, feedback, messages):
    turn3_user = TURN3_REFINE_TEMPLATE.format(
        question=question_text,
        prev_response=resp1,
        feedback=feedback,
    )
    messages.append({"role": "user", "content": turn3_user})
    return calculate_input_tokens(messages), messages   


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
    
    tt_num = 0
    output_len = 0
    total = len(data)
    input_len = 0

    if isbbq:
        
        with open("<REPO_ROOT>/datasets/bbq/processed/test/test.jsonl", 'r') as f:
            original_file = f.readlines()
        
        original_file = [json.loads(line) for line in original_file]
        original_file = {itm["extra_info"]["uuid"]: itm["extra_info"]["context_condition"] for itm in original_file}
        
        
        for itm in data:
            
            output = itm["turn2_feedback"]
            output_len += get_output_length(output)
            output = itm["turn3_response"]
            output_len += get_output_length(output)
            
            input_len2, messages = turn_2_input_len(itm["initial_reasoning"], itm["raw_prompt"])
            input_len += input_len2
            
            input_len3, messages = turn_3_input_len(itm["raw_prompt"], itm["initial_reasoning"], itm["turn2_feedback"], messages)
            input_len += input_len3
            
            
            if original_file[itm["uuid"]] == "ambig":
                if itm["pred"] == itm["gold"]:
                    correct_num += 1
                tt_num += 1
        
        print(f"Acc: {correct_num/tt_num:2f}; Total: {tt_num}")
        print(f"Overall input len: {input_len}/{total} = {input_len / total:.2f}\n")
        print(f"Overall output len: {output_len}/{total} = {output_len / total:.2f}\n")
        
    else:
        for itm in data:
            output = itm["turn2_feedback"]
            output_len += get_output_length(output)
            output = itm["turn3_response"]
            output_len += get_output_length(output)
            
            input_len2, messages = turn_2_input_len(itm["initial_reasoning"], itm["raw_prompt"])
            input_len += input_len2
            
            input_len3, messages = turn_3_input_len(itm["raw_prompt"], itm["initial_reasoning"], itm["turn2_feedback"], messages)
            input_len += input_len3
            
            if itm["pred"] == itm["gold"]:
                correct_num += 1
        
        print(f"Acc: {correct_num/len(data):2f}; Total: {len(data)}")
        print(f"Overall input len: {input_len}/{total} = {input_len / total:.2f}\n")
        print(f"Overall output len: {output_len}/{total} = {output_len / total:.2f}\n")


if __name__ == "__main__":
    
    for model in ["llama3.1-8b-instruct"]:
        # , "qwen2.5-7b-instruct"]:
        for dataset in ["bbq", "crowdspairs", "genMO", "stereoset", "winoqueer"]:
            
            print(model, dataset)
            pred_file = f"<REPO_ROOT>/results/original.unbiased_messages/{model}/{dataset}.original.{model}.iasc.jsonl"
            print(pred_file)
            
            calc_answer(pred_file, isbbq=(dataset == "bbq"))
            