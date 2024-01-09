import argparse
import os
import random
import json
from benchmark.analyse import env2datasets, ENVS, model2maxlen, dataset2level
from environment.aminer_run.agent_arch import f1_score

dataset2env = {
    "high_freq_ent": "wiki",
    "low_freq_ent": "wiki",
    "csj": "wiki",
    "cpj": "wiki",
    "cic": "wiki",
    "hotpotqa": "wiki",
    "2wikimultihopqa": "wiki",
    "musique": "wiki",
    "kqapro": "wiki",
    "soay_easy": "aminer",
    "profiling": "aminer",
    "soay_hard": "aminer"
}
LLM = ["llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"]
AGENT = ["React", "chatgpt_function", "PAL", "dfsdt"]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="single", choices=["load_all", "single"])
    # environment args
    parser.add_argument('--environment', type=str, default="wiki", choices=["wiki", "aminer"])
    # agent args
    parser.add_argument('--models', type=str, default=["llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"])
    parser.add_argument('--model', type=str, default="gpt-4-1106-preview", choices=LLM)
    parser.add_argument('--agent_name', type=str, default="dfsdt", choices=AGENT)
    return parser.parse_args(args)

def get_retrieved_passages(train_messages):
    prompt = []
    scratchpad = []
    for messages in train_messages:
        for message in messages:
            if message["role"] == "system" or message["role"] == "user":
                prompt.append(message["content"])
            else:
                scratchpad.append(message["content"])
    return " ".join(prompt), " ".join(scratchpad)

def load(model, agent_name):
    # get raw question and answer
    raw_data_dict = {}
    for env in ENVS:
        for level in ["easy", "medium", "hard"]:
            dataset_names = env2datasets[env][level]
            for dataset in dataset_names:
                raw_data = []
                with open("data/KoLA2/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
                    for line in f:
                        raw_data.append(json.loads(line)) 
                raw_data_dict[dataset] = raw_data
    # get other agents' results
    results = {}
    if agent_name == "PAL":
        root_dir = "/home/ubuntu/soay-wiki/results"
    else:
        root_dir = "/home/ubuntu/xyy_use/bench/data/answer/kolaans"
        sub_name = f"{model}_{agent_name}"
        data_dir = os.path.join(root_dir, sub_name)
        sub_dir_names = os.listdir(data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir = os.path.join(data_dir, sub_dir_name)
            if os.path.isdir(sub_dir):
                res = []
                data_set_name = sub_dir_name[4:]
                raw_data = raw_data_dict[data_set_name]
                for i, json_obj in enumerate(raw_data):
                    _id = 20000 + i
                    file_name = f"{_id}_DFS_woFilter_w2.json"
                    file_path = os.path.join(sub_dir, file_name)
                    question = json_obj["input"]
                    answer = json_obj["outputs"][0]
                    with open(file_path, "r") as f:
                        raw_res = json.load(f)
                        # check the answer_generation key
                        answer_generation = raw_res["answer_generation"]
                        final_answer_idct = json.loads(answer_generation["final_answer"])
                        if "final_answer" in final_answer_idct:
                            prediction = final_answer_idct["final_answer"]
                        else:
                            prediction = "give up"
                        query = answer_generation['query']
                        prompt, scratchpad = get_retrieved_passages(answer_generation['train_messages'])
                        assert query == question
                        f1 = f1_score(prediction, answer)
                        is_corre = True if prediction == answer else False
                        _dict = {
                            "question": question,
                            "answer": answer,
                            "prediction": prediction,
                            "prompt": prompt,
                            "scratchpad": scratchpad,
                            "reward": f1,
                            "correct": is_corre,
                            "halted": False, 
                            "error": False, 
                        }
                        res.append(_dict)
                
                results[data_set_name] = res
    return results



def get_res(model, agent_name):
    # step 1 : load the results
    results = load(model, agent_name)
    print("Full results on datasets:")
    print(len(results))
    print(list(results.keys()))
    # step 2 : save the results
    for data_set_name, result_list in results.items():
        save_dir = f"data/result/{agent_name}_{dataset2env[data_set_name]}_run_Agent_{model}_{model2maxlen[model]}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{data_set_name}_log.jsonl")
        with open(save_path, "w") as f:
            for result in result_list:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    args = parse_args()
    if args.task == "load_all":
        for model in args.models:
            get_res(model, args.agent_name)
    else:
        get_res(args.model, args.agent_name)