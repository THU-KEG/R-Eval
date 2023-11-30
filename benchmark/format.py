import argparse
import joblib
import os
import json
import hashlib

dataset2level = json.load(open("config/dataset2level.json", "r"))
origin_datasets = ["hotpotqa"]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # environment args
    parser.add_argument('--environment', type=str, default="wiki", choices=["wiki", "aminer"])
    parser.add_argument( # for specific dataset
        "--dataset",
        type=str,
        default="all",
        choices=["all", "hotpotqa","lcc", "multi_news", "qmsum","alpacafarm"],
    )
    return parser.parse_args(args)


def convert_to_sha256(string):
    # Encoding the string into bytes
    encoded_string = string.encode('utf-8')

    # Creating a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Updating the hash object with the encoded string
    sha256_hash.update(encoded_string)

    # Obtaining the hexadecimal representation of the hash
    hex_digest = sha256_hash.hexdigest()

    return hex_digest

def word_tokenize(text):
    return text.split()


def cal_len_and_output(_input, output, f, dataset, env):
    # calculate length
    input_count = len(word_tokenize(_input))
    if type(output) == list:
        output_count =  sum([ len(word_tokenize(o)) for o in output]) / len(output)
    else:
        output_count = len(word_tokenize(output))
    json_obj = {}
    json_obj["input"] = _input # The input/command for the task, usually short, such as questions in QA, queries in Few-shot tasks, etc
    if type(output) == list:
        json_obj["outputs"] = output
    else:
        json_obj["outputs"] = [output]
    json_obj["input_length"] = input_count
    json_obj["output_length"] = output_count
    # other keys
    json_obj["language"] = "en"
    json_obj["environment"] = env
    json_obj["dataset"] = dataset
    encode_str = "Tsinghua KoLA2 " + json.dumps(json_obj)
    json_obj["_id"] = convert_to_sha256(encode_str)
    json.dump(json_obj, f)
    f.write("\n")
  


def load_wiki_hotpotqa(output_dir, dataset):
    hotpot = joblib.load(f'data/raw/hotpotqa/medium.joblib').reset_index(drop = True)
    task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]
    output_path = f"{output_dir}/{dataset2level[dataset]}_{dataset}.jsonl"
    with open(output_path, "w") as f:
        for _instance in task_instructions:
            input = _instance[0]
            output = _instance[1]
            cal_len_and_output(input, output, f, dataset, env="wiki")



def get_data(environment, dataset):
    output_dir = 'data/KoLA2'
    if environment == "wiki":
        if dataset == "hotpotqa":
            load_wiki_hotpotqa(output_dir, dataset)
            

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "all":
        for origin_dataset in origin_datasets:
            get_data(args.environment, origin_dataset)
    else:
        get_data(args.environment, args.dataset)