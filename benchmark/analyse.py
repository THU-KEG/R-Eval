import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pred import seed_everything
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
dataset2level = json.load(open("config/dataset2level.json", "r"))
env2datasets = {
    "wiki" : {
        "easy": ["high_freq_ent", "low_freq_ent"],
        "medium": ["csj", "cpj", "cic"],
        "hard": ["hotpotqa", "2wikimultihopqa", "musique", "kqapro"]
    },
    "aminer" :{
        "easy": ["soay_easy"],
        "medium": ["profiling"],
        "hard": ["soay_hard"]
    }
}
LLM = ["llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"]
# AGENT = ["React", "FC", "PAL", "DFSDT"]
AGENT = ["React"]
ENVS = ["wiki", "aminer"]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # environment args
    parser.add_argument('--result_dir', type=str, default="data/result")
    parser.add_argument('--out_dir', type=str, default="data/analysis")
    # parser.add_argument('--environment', type=str, default="wiki", choices=ENVS)
    # agent args
    parser.add_argument('--model', type=str, default="tulu-7b", choices=LLM)
    parser.add_argument('--agent_name', type=str, default="React", choices=AGENT)
    parser.add_argument('--comp_model', type=str, default="llama2-7b-chat-4k", choices=LLM)
    parser.add_argument('--comp_agent_name', type=str, default="React", choices=AGENT)
    # analyse args
    parser.add_argument('--aspect', type=str, default="combination", choices=["retrieval", "model", "combination"])
    parser.add_argument('--function', type=str, default="all", choices=["single", "pairwise", "all"])  
    return parser.parse_args(args)


class Analyzer:
    def __init__(self, args) -> None:
        self.args = args
        self.aspect = args.aspect
        self.function = args.function
        # load data
        self.data = self.load_data(args)
        # define output dir
        exp_name = f"{args.function}_{args.aspect}_{args.agent_name}_{args.model}_{args.comp_agent_name}_{args.comp_model}"
        self.output_dir = os.path.join(args.out_dir, exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_single_data(self, agent_name, model_name):
        data_dict = {
            "agent_name": agent_name,
            "model_name": model_name,
        }
        for env in ENVS:
            env_dict = {}
            max_length = model2maxlen[model_name]
            dir_name = f"{agent_name}_{env}_run_Agent_{model_name}_{max_length}"
            real_data_dir = os.path.join(self.args.result_dir, dir_name)
            # under this dir there are many jsonl files, we load by difficulty
            datasets = env2datasets[env]
            for difficulty, dataset_list in datasets.items():
                data_lst = []
                for dataset in dataset_list:
                    jsonl_file = os.path.join(real_data_dir, f"{dataset}_log.jsonl")
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            json_obj = json.loads(line)
                            data_lst.append(json_obj)
                # sort the data_lst by question
                data_lst = sorted(data_lst, key=lambda x: x["question"])
                env_dict[difficulty] = data_lst
            data_dict[env] = env_dict
        return data_dict

    def load_pairwise_data(self, args):
        data_a  = self.load_single_data(args.agent_name, args.model)
        data_b = self.load_single_data(args.comp_agent_name, args.comp_model)
        data = [ data_a, data_b ]
        return data

    def load_data(self, args):
        if self.function == "single":
            data = self.load_single_data(args.agent_name, args.model)
            # will get 200 300 400 json data
            # print(len(data["wiki"]["easy"]))
            # print(len(data["wiki"]["medium"]))
            # print(len(data["wiki"]["hard"]))
            data = [data]
            
        elif self.function == "pairwise":
            data = self.load_pairwise_data(args)
        else:
            # get all data
            data = []
            for agent_name in AGENT:
                for model_name in LLM:
                    data.append(self.load_single_data(agent_name, model_name))
        # this is a list of dict, each dict is a system's logs on an env, each env has 3 difficulty
        return data
    
    def draw_histogram(self, df, y_label, title):
        plt.rc('font',family='Times New Roman')
        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=df, kind="bar",
            x="env", y="reward", hue="difficulty",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("Domain", y_label)
        # set the title of picture
        g.fig.suptitle(title)
        out_path = os.path.join(self.output_dir, f"{title}_histogram.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    def retrieval_analysis(self):
        pass

    def model_analysis(self):
        pass

    def combination_analysis(self):
        # step1: draw performance histogram
        best_system = self.data[0]
        for data_dict in self.data:
            final_data = []
            system_name = f"{data_dict['agent_name']}_{data_dict['model_name']}"
            for env in ENVS:
                for difficulty, data_lst in data_dict[env].items():
                    reward_lst = [json_obj["reward"] for json_obj in data_lst]
                    avg_reward = sum(reward_lst) / len(reward_lst)
                    final_dict = {
                        "env": env,
                        "difficulty": difficulty,
                        "reward": avg_reward,
                    }
                    final_data.append(final_dict)
                    # step2: get best of n system for combination score
                    for i, reward in enumerate(reward_lst):
                        last_reward = best_system[env][difficulty][i]["reward"]
                        if reward > last_reward:
                            best_system[env][difficulty][i] = data_lst[i]
            df = pd.DataFrame(final_data)
            self.draw_histogram(df, y_label="F1", title=system_name)
        # step3: draw best system histogram
        final_data = []
        for env in ENVS:
            for difficulty, data_lst in best_system[env].items():
                reward_lst = [json_obj["reward"] for json_obj in data_lst]
                avg_reward = sum(reward_lst) / len(reward_lst)
                final_dict = {
                    "env": env,
                    "difficulty": difficulty,
                    "reward": avg_reward,
                }
                final_data.append(final_dict)
        df = pd.DataFrame(final_data)
        self.draw_histogram(df, y_label="F1", title="best_system")
        

    def run(self):
        if self.aspect == "retrieval":
            self.retrieval_analysis()
        elif self.aspect == "model":
            self.model_analysis()
        else:
            self.combination_analysis()


if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)
    analyzer = Analyzer(args)
    analyzer.run()
    