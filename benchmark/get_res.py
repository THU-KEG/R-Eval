import argparse
import os
from tqdm import tqdm
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
            if message["content"] is not None:
                if message["role"] == "system" or message["role"] == "user":
                    prompt.append(message["content"])
                else:
                    scratchpad.append(message["content"])
    return " ".join(prompt), " ".join(scratchpad)


def load_raw_data_dict():
    raw_data_dict = {}
    for env in ENVS:
        for level in ["KM", "KU", "KA"]:
            dataset_names = env2datasets[env][level]
            for dataset in dataset_names:
                raw_data = []
                with open("data/KoLA2/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
                    for line in f:
                        raw_data.append(json.loads(line)) 
                raw_data_dict[dataset] = raw_data
    return raw_data_dict

def load(model, agent_name):
    # get raw question and answer
    raw_data_dict = load_raw_data_dict()
    # get other agents' results
    results = {}
    if agent_name == "PAL":
        root_dir = "/home/ubuntu/soay-wiki/results"
        sub_name = f"{model}"
        data_dir = os.path.join(root_dir, sub_name)
        jsonl_names = os.listdir(data_dir)
        for jsonl_name in jsonl_names:
            if jsonl_name.endswith("jsonl"):
                res = []
                data_set_name = jsonl_name.split(".")[0][4:]
                print(data_set_name)
                if data_set_name in ["cqa"]:
                    continue
                jsonl_path = os.path.join(data_dir, jsonl_name)
                with open(jsonl_path, "r") as f:
                    for line in f.readlines():
                        json_obj = json.loads(line)
                        question = json_obj["query_info"]["input"]
                        answer = json_obj["query_info"]["outputs"][0]
                        prediction = json_obj["answer"]
                        f1 = f1_score(prediction, answer)[0]
                        is_corre = True if prediction == answer else False
                        _dict = {
                            "question": question,
                            "answer": answer,
                            "prediction": prediction,
                            "prompt": json_obj["code"],
                            "scratchpad": json_obj["info"],
                            "reward": f1,
                            "correct": is_corre,
                            "halted": False, 
                            "error": False, 
                        }
                        res.append(_dict)
                results[data_set_name] = res
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
                print(data_set_name)
                if data_set_name in ["cqa"]:
                    continue
                try:
                    raw_data = raw_data_dict[data_set_name]
                except KeyError:
                    print("KeyError")
                    continue
                for i, json_obj in tqdm(enumerate(raw_data)):
                    _id = 20000 + i
                    file_name = f"{_id}_DFS_woFilter_w2.json"
                    file_path = os.path.join(sub_dir, file_name)
                    if not os.path.exists(file_path):
                        print("Lack instance", file_path)
                        continue
                    question = json_obj["input"]
                    answer = json_obj["outputs"][0]
                    with open(file_path, "r") as f:
                        raw_res = json.load(f)
                        # check the answer_generation key
                        answer_generation = raw_res["answer_generation"]
                        try:
                            final_answer_idct = json.loads(answer_generation["final_answer"])
                        except json.JSONDecodeError:
                            final_answer_idct = {
                                "final_answer": ""
                            }
                            answer_generation['train_messages']  =  []
                        if "final_answer" in final_answer_idct:
                            prediction = final_answer_idct["final_answer"]
                        else:
                            prediction = "give up"
                        query = answer_generation['query']
                        try:
                            prompt, scratchpad = get_retrieved_passages(answer_generation['train_messages'])
                        except TypeError:
                            print("TypeError")
                            # print(answer_generation['train_messages'])
                            quit()
                        if query != question:
                            print("query != question")
                            question = query
                            # print(query)
                            # print("question:")
                            # print(question)
                            continue
                        f1 = f1_score(prediction, answer)[0]
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