import os
import json
from tqdm import tqdm
import numpy as np
import random
import joblib
import concurrent
import environment.wiki_run.utils as utils
from environment.wiki_run.llms import get_llm_backend
from environment.wiki_run.agent_arch import get_wiki_agent
from environment.aminer_run.agent_arch import get_aminer_agent
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # agent args
    parser.add_argument('--model', type=str, default="tulu-7b", choices=["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"])
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
    parser.add_argument('--num_workers', type=int, default=1) # for multi-threading, suitable for api-based llms like gpt3.5
    return parser.parse_args(args)


def process_agent_run_step(agent):
    agent.run()

def get_pred(args, data, max_context_length, dataset_name, llm_name, save_dir):
    num_workers = args.num_workers
    task_instructions = [(json_obj["input"],json_obj["outputs"][0]) for json_obj in data]
    # TODO: remove this htotpotqa hard code
    hotpot = joblib.load(f'data/raw/hotpotqa/medium.joblib').reset_index(drop = True)
    task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]
    if args.environment == "wiki":
        agent_cls = get_wiki_agent(args.agent_name, dataset_name)
    elif args.environment == "aminer":
        agent_cls = get_aminer_agent(args.agent_name, dataset_name)
    agent_save_file = os.path.join(save_dir, f"{dataset_name}_log.jsonl")
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        print(f"{dataset_name} finished:{len(completed_tasks)}")
        task_instructions = [task for task in task_instructions if task not in completed_tasks]
        utils.delete_error(agent_save_file)
    llm = get_llm_backend(llm_name).run
    agents = [agent_cls(ques, ans, llm, max_context_length) for ques, ans in task_instructions]
    if num_workers <= 1:
        for agent in tqdm(agents, total=len(agents)):
            agent.run()
            utils.log_agent(agent, agent_save_file)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(process_agent_run_step, agents)
        for agent in tqdm(agents, total=len(agents)):
            utils.log_agent(agent, agent_save_file)
    print(f'Finished Trial. Total: {len(agents)}')

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def run_dataset(dataset):
    print(f"{dataset} has began.........")
    data = []
    with open("data/raw/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))       
    get_pred(args, data, max_length, dataset, model_name, save_dir)
                    

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
            run_dataset(dataset)
    else:
        dataset = args.dataset
        run_dataset(dataset)
        