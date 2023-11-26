import os
from datasets import load_dataset
import torch, gc
import json
from tqdm import tqdm
import numpy as np
import random
from environment.wiki_run.llms import get_llm_backend
from environment.wiki_run.agent_arch import get_wiki_agent
from environment.aminer_run.agent_arch import get_aminer_agent
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # agent args
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-1106", choices=["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "internlm-7b-8k", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
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


def get_pred(args, data, max_context_length, dataset_name, llm_name):
    preds = []
    if args.environment == "wiki":
        agent_cls = get_wiki_agent(args.agent_name, dataset_name)
    elif args.environment == "aminer":
        agent_cls = get_aminer_agent(args.agent_name, dataset_name)
    
    for json_obj in tqdm(data[args.start_point:]):
        # for gpt-3.5 and gpt-4, we use the prompt without tokenization
        ques = json_obj["input"]
        ans = json_obj["outputs"][0]
        llm = get_llm_backend(llm_name).run
        agent = agent_cls(ques, ans, llm, max_context_length) 
        completions_text  = agent.run()
        preds.append({"input":ques, "pred": completions_text, "answers": json_obj["outputs"], "all_classes": json_obj["all_classes"], "length":json_obj["length"]})
        
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    max_length = model2maxlen[model_name]
    if args.environment == "wiki":
        datasets = ["multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "passage_retrieval_en"]
    else:
        datasets = ["longform_qa", "finance_qa","hotpotqa","lcc", "multi_news", "qmsum","alpacafarm"]
    # id for each dataset
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
            preds = get_pred(args, data, max_length, dataset, model_name)
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
        preds = get_pred(args, data, max_length, dataset, model_name)
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
                    
                    