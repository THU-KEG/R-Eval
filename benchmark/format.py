import argparse
import joblib
import os
import random
import json
import hashlib
from pred import seed_everything



dataset2level = json.load(open("config/dataset2level.json", "r"))
origin_datasets = ["hotpotqa"]
level2kola_path = {
    "1-1": "/data2/cookie/input/KG/2_high_freq_ent/test.json",
    "1-2": "/data2/cookie/input/KG/1_low_freq_ent/test.json",
    "1-3": "/home/ubuntu/KoLA2/data/raw/soayBench_v1-2",
    "2-1": "/data2/cookie/input/IE/COPEN/csj/dev.json",
    "2-2": "/data2/cookie/input/IE/COPEN/cpj/dev.json",
    "2-3": "/data2/cookie/input/IE/COPEN/cic/dev.json",
    "2-4": "/home/ubuntu/KoLA2/data/raw/profiling/test.json",
    "3-1": "hotpotqa",
    "3-2": "/home/ubuntu/KoLA2/data/raw/up/2wiki_dev.json",
    "3-3": "/home/ubuntu/KoLA2/data/raw/up/musique_ans_v1.0_dev.jsonl",
    "3-4": "/home/ubuntu/KoLA2/data/raw/kqapro/val.json",
    "3-5": "/home/ubuntu/KoLA2/data/raw/soay/data.jsonl",
    "3-6": "/home/ubuntu/KoLA2/data/raw/soayBench_v1-2",
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # environment args
    parser.add_argument('--environment', type=str, default="wiki", choices=["wiki", "aminer"])
    parser.add_argument( # for specific dataset
        "--dataset",
        type=str,
        default="kola",
        choices=["all", "hotpotqa","kola", "kqapro", "cqa", "profiling","soay"],
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

def load_kola(output_dir, dataset):
    # need 1-1, 1-2. COPEN
    # datasets = [ "high_freq_ent", "low_freq_ent", "csj", "cpj", "cic", "2wikimultihopqa", "musique"]
    # datasets = [ "musique", "2wikimultihopqa"]
    datasets = [ "musique"]
    for dataset in datasets:
        level = dataset2level[dataset]
        data_path = level2kola_path[level]
        output_path = f"{output_dir}/{level}_{dataset}.jsonl"
        with open(output_path, "w") as f:
            if dataset == "2wikimultihopqa":
                data_file = json.load(open(data_path, 'r'))
                data_file = random.sample(data_file, 100)
                print(len(data_file))
                for _instance in data_file:
                    question = _instance['question']
                    answer = _instance['answer']
                    cal_len_and_output(question, answer, f, dataset, env="wiki")
            elif dataset == "musique":
                fin = open(data_path, 'r')
                lines = fin.readlines()
                candidates = []
                fin = open(data_path, 'r')
                lines = fin.readlines()
                candidates = []
                questions = []
                for line in lines:
                    _instance = json.loads(line.strip())
                    question = _instance['question']
                    if question in questions:
                        continue
                    else:
                        questions.append(question)
                    answer = _instance['answer']
                    if _instance['answerable']:
                        candidates.append([question, answer])
                data_file = random.sample(candidates, 100)
                print(len(data_file))
                for _instance in data_file:
                    question = _instance[0]
                    answer = _instance[1]
                    cal_len_and_output(question, answer, f, dataset, env="wiki")
            else:
                data_file = json.load(open(data_path, 'r'))["request_states"]
                for _instance in data_file:
                    instance = _instance["instance"]
                    input = instance["input"]["text"]
                    output = instance["references"][0]["output"]["text"]
                    cal_len_and_output(input, output, f, dataset, env="wiki")

def load_kqapro(output_dir, dataset):
    level = dataset2level[dataset]
    data_path = level2kola_path[level] 
    output_path = f"{output_dir}/{level}_{dataset}.jsonl"
    with open(output_path, "w") as f:
        data_file = json.load(open(data_path, 'r'))
        data_file = random.sample(data_file, 100)
        print(len(data_file))
        for _instance in data_file:
            question = _instance['question']
            answer = _instance['answer']
            cal_len_and_output(question, answer, f, dataset, env="wiki")


def load_soay(output_dir, dataset):
    level = dataset2level[dataset]
    data_path = level2kola_path[level] 
    output_path = f"{output_dir}/{level}_{dataset}.jsonl"
    with open(output_path, "w") as f:
        lines = open(data_path, 'r').readlines()
        data_file = []
        for line in lines:
            _instance = json.loads(line.strip())
            question = _instance['Query_EN']
            answer = _instance["Answer"]
            if answer is not str:
                try:
                    answer = str(answer)
                except TypeError:
                    answer = json.dumps(answer)
            data_file.append({"question": question, "answer": answer})
        data_file = random.sample(data_file, 100)
        print(len(data_file))
        for _instance in data_file:
            question = _instance['question']
            answer = _instance['answer']
            cal_len_and_output(question, answer, f, dataset, env="aminer")

def load_profiling(output_dir, dataset):
    final_distribution =  {'Professor': 20, 'Other': 16, 'Researcher': 14, 'Associate Professor': 14, 'Assistant Professor': 10, 'Engineer': 10, 'Lecturer': 6, 'Associate Researcher': 3, 'Ph.D': 3, 'Senior Engineer': 2, 'Assistant Researcher': 2}
    level = dataset2level[dataset]
    data_path = level2kola_path[level] 
    output_path = f"{output_dir}/{level}_{dataset}.jsonl"
    with open(output_path, "w") as f:
        data_file = json.load(open(data_path, 'r'))
        # data_file = random.sample(data_file, 1700)
        print(len(data_file))
        # answers = []
        title2record = {}
        for _instance in data_file:
            title_text = _instance['title_text']
            org = _instance['org']
            name = _instance['name']
            question = f"Your task is to predict a person's position, please choose from: Professor, Researcher, Associate Professor, Assistant Professor, Lecturer, Engineer, Senior Engineer, Ph.D, Associate Researcher, Assistant Researcher, Other. Given search engine records: {title_text}, please identify the position of {name} from {org}."
            answer = _instance['title'].split("(")[0].strip()
            try:
                title2record[answer].append((question, answer))
            except KeyError:
                title2record[answer] = [(question, answer)]
            # answers.append(answer)
            # cal_len_and_output(question, answer, f, dataset, env="aminer")
        # count distribution
        # from collections import Counter
        # counter = Counter(answers)
        # print(counter)
        for k, v in final_distribution.items():
            print(k, v)
            if k not in title2record:
                continue
            record = title2record[k]
            # sample
            record = random.sample(record, v)
            for _instance in record:
                question = _instance[0]
                answer = _instance[1]
                cal_len_and_output(question, answer, f, dataset, env="aminer")

def load_soay_v1(output_dir, dataset):
    # read data
    data_dir = level2kola_path['1-3'] 
    hardness2data = {
        'easy': [],
        'hard': []
    }
    for file_number in range(1, 45):
        file_name = f"{file_number}.jsonl"
        data_path = os.path.join(data_dir, file_name)
        lines = open(data_path, 'r').readlines()
        if file_number < 8:
            hardness = 'easy'
        else:
            hardness = 'hard'
        for i, line in enumerate(lines):
            _instance = json.loads(line.strip())
            question = _instance['Query_en']
            answer = _instance["Answer"]
            if answer is not str:
                try:
                    answer = str(answer)
                except TypeError:
                    answer = json.dumps(answer)
            hardness2data[hardness].append({"question": question, "answer": answer})
    # output
    for level in ['1-3', '3-5']:
        if level == '1-3':
            dataset_hardness = 'easy'
        else:
            dataset_hardness = 'hard'
        output_path = f"{output_dir}/{level}_soay_{dataset_hardness}.jsonl"
        with open(output_path, "w") as f:
            data_file = hardness2data[dataset_hardness]
            data_file = random.sample(data_file, 100)
            print(len(data_file))
            for _instance in data_file:
                question = _instance['question']
                answer = _instance['answer']
                cal_len_and_output(question, answer, f, dataset, env="aminer")


def get_data(environment, dataset):
    output_dir = 'data/KoLA2'
    if environment == "wiki":
        if dataset == "hotpotqa":
            load_wiki_hotpotqa(output_dir, dataset)
        elif dataset == "kola":
            load_kola(output_dir, dataset)
        elif dataset == "kqapro":
            load_kqapro(output_dir, dataset)
    elif environment == "aminer":
        if dataset == "cqa":
            load_soay(output_dir, dataset)
        elif dataset == "profiling":
            load_profiling(output_dir, dataset)
        elif dataset == "soay":
            load_soay_v1(output_dir, dataset)
            

if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)
    if args.dataset == "all":
        for origin_dataset in origin_datasets:
            get_data(args.environment, origin_dataset)
    else:
        get_data(args.environment, args.dataset)