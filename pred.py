import os
from datasets import load_dataset
import torch, gc
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
from wiki_run.agent_arch import get_wiki_agent
from aminer_run.agent_arch import get_aminer_agent
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # agent args
    parser.add_argument('--model', type=str, default="llama2-7b-chat-4k", choices=["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "internlm-7b-8k", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
    parser.add_argument('--agent_name', type=str, default="React_wiki_run_Agent", choices=["Zeroshot_wiki_run_Agent", "React_wiki_run_Agent", "Planner_wiki_run_Agent", "PlannerReact_wiki_run_Agent"])
    # environment args
    parser.add_argument('--environment', type=str, default="wiki", choices=["wiki", "aminer"])
    parser.add_argument( # for specific dataset
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=["konwledge_memorization","konwledge_understanding","longform_qa",
                        "finance_qa","hotpotqa","lcc", "multi_news", "qmsum","alpacafarm", "all"],
    )
    parser.add_argument('--start_point', type=int, default=0)
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)      
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "tulu" in model_name:
        prompt = f"<|user|>:{prompt}\n<|assistant|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(args, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, debug: bool = False):
    preds = []
    if args.environment == "wiki":
        agent_cls = get_wiki_agent(args.agent_name)
    elif args.environment == "aminer":
        agent_cls = get_aminer_agent(args.agent_name)
    
    torch.cuda.empty_cache()
    for json_obj in tqdm(data[args.start_point:]):
        prompt = prompt_format.format(**json_obj)
        if tokenizer:
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
            ques = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            # for gpt-3.5 and gpt-4, we use the prompt without tokenization
            ques = prompt
        ans = json_obj["outputs"][0]
        agent = agent_cls(ques, ans, model, max_length) 

        completions_text, completions_tokens  = agent(input_ids=ques.input_ids, max_new_tokens=max_gen)
        
        if debug:
            print("####################")
            
        pred = completions_text
        pred = post_process(pred, model_name)
        preds.append({"prompt":prompt, "pred": pred, "completions_tokens":completions_tokens, "answers": json_obj["outputs"], "all_classes": json_obj["all_classes"], "length":json_obj["length"]})
        
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device,  load_token_only=False):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if not load_token_only:
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                  output_scores=True, return_dict_in_generate=True, 
                                                  torch_dtype=torch.bfloat16).to(device)
            model.eval()
    elif "llama2" or "tulu" in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        if not load_token_only:
            model = LlamaForCausalLM.from_pretrained(path, output_scores=True, return_dict_in_generate=True, 
                                                 torch_dtype=torch.bfloat16).to(device) 
    elif "gpt-3.5" or "gpt-4" in model_name:
        return None, None
    
    if load_token_only:
        return tokenizer
    else:
        model = model.eval()
        return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = args.model
    # define your model
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]
    if args.environment == "wiki":
        datasets = ["multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "passage_retrieval_en"]
    else:
        datasets = ["longform_qa", "finance_qa","hotpotqa","lcc", "multi_news", "qmsum","alpacafarm"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    dataset2level = json.load(open("config/dataset2level.json", "r"))
    # make dir for saving predictions
    if not os.path.exists("data/result"):
        os.makedirs("data/result")
    save_dir = f"data/result/{args.agent_name}_{model_name}_{max_length}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # predict on each dataset
    if args.dataset == "all":
        for dataset in datasets:
            # load data
            print(f"{dataset} has began.........")
            data = []
            with open("data/raw/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))       
            out_path = os.path.join(save_dir, f"{dataset}.jsonl")
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            preds = get_pred(args, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
                    
    else:
        dataset = args.dataset
        print(f"{dataset} has began.........")
        data = []
        with open("data/raw/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))       
        out_path = os.path.join(save_dir, f"{dataset}.jsonl")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(args, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        if os.path.exists(out_path):
            with open(out_path, "a", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
                    
                    