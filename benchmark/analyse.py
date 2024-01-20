import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import textwrap
from itertools import combinations
import sys
sys.path.append('/home/ubuntu/KoLA2')
from pred import seed_everything
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
dataset2level = json.load(open("config/dataset2level.json", "r"))
# env2datasets = {
#     "wiki" : {
#         "easy": ["high_freq_ent", "low_freq_ent"],
#         "medium": ["csj", "cpj", "cic"],
#         "hard": ["hotpotqa", "2wikimultihopqa", "musique", "kqapro"]
#     },
#     "aminer" :{
#         "easy": ["soay_easy"],
#         "medium": ["profiling"],
#         "hard": ["soay_hard"]
#     }
# }
env2datasets = {
    "wiki" : {
        "KM": ["high_freq_ent", "low_freq_ent"],
        "KU": ["csj", "cpj", "cic"],
        "KA": ["hotpotqa", "2wikimultihopqa", "musique", "kqapro"]
    },
    "aminer" :{
        "KM": ["soay_easy"],
        "KU": ["profiling"],
        "KA": ["soay_hard"]
    }
}
LLM = ["llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"]
llm2abc = {
    "llama2-7b-chat-4k": 'A', 
    "tulu-7b": 'B', 
    "llama2-13b": 'C',
    "vicuna-13b": 'D', 
    "gpt-3.5-turbo-1106": 'F', 
    "gpt-4-1106-preview": 'G', 
    "codellama-13b-instruct": 'H', 
    "toolllama-2-7b": 'I'
}
# AGENT = ["React", "FC", "PAL", "DFSDT"]
AGENT = ["React"]
ENVS = ["wiki", "aminer"]
DIFF = ["KM", "KU", "KA"]

def get_all_keys(d):
    all_keys = set()

    def _get_keys(data, prefix=""):
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key
            all_keys.add(current_key)

            if isinstance(value, dict):
                _get_keys(value, prefix=current_key)

    _get_keys(d)
    return all_keys

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


class DataLoader:
    def __init__(self, result_dir, function, agent_name, model, comp_agent_name, comp_model) -> None:
        self.result_dir = result_dir
        self.function = function
        self.data = self.load_data()
        self.agent_name = agent_name
        self.model = model
        self.comp_agent_name = comp_agent_name
        self.comp_model = comp_model

    def load_single_data(self, agent_name, model_name):
        data_dict = {
            "agent_name": agent_name,
            "model_name": model_name,
        }
        for env in ENVS:
            env_dict = {}
            max_length = model2maxlen[model_name]
            dir_name = f"{agent_name}_{env}_run_Agent_{model_name}_{max_length}"
            real_data_dir = os.path.join(self.result_dir, dir_name)
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

    def load_pairwise_data(self):
        data_a  = self.load_single_data(self.agent_name, self.model)
        data_b = self.load_single_data(self.comp_agent_name, self.comp_model)
        data = [ data_a, data_b ]
        return data

    def load_data(self):
        if self.function == "single":
            data = self.load_single_data(self.agent_name, self.model)
            # will get 200 300 400 json data
            # print(len(data["wiki"]["easy"]))
            # print(len(data["wiki"]["medium"]))
            # print(len(data["wiki"]["hard"]))
            data = [data]
            
        elif self.function == "pairwise":
            data = self.load_pairwise_data()
        else:
            # get all data
            data = []
            for agent_name in AGENT:
                for model_name in LLM:
                    data.append(self.load_single_data(agent_name, model_name))
        # this is a list of dict, each dict is a system's logs on an env, each env has 3 difficulty
        return data
   
class Analyzer:
    def __init__(self, args) -> None:
        self.args = args
        self.aspect = args.aspect
        self.function = args.function
        # load data
        data_loader = DataLoader(args.result_dir, args.function, args.agent_name, 
                                 args.model, args.comp_agent_name, args.comp_model) 
        self.data = data_loader.data
        # define output dir
        exp_name = f"{args.function}_{args.aspect}_{args.agent_name}_{args.model}_{args.comp_agent_name}_{args.comp_model}"
        self.output_dir = os.path.join(args.out_dir, exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

     
    def draw_histogram(self, df, y_label, title):
        plt.rc('font',family='Times New Roman')
        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=df, kind="bar",
            x="env", y="reward", hue="difficulty",
            errorbar='sd', palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("Domain", y_label)
        # set the title of picture
        g.fig.suptitle(title)
        out_path = os.path.join(self.output_dir, f"{title}_histogram.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    def draw_boxplot(self, df, y_label, title):
        # step1: set basic info
        plt.rc('font',family='Times New Roman')
        sns.set_theme(style="ticks")
        b=sns.boxplot(
            data=df, x="difficulty", y="reward", hue="env",
            palette="flare"
        )
        sns.despine(offset=10, top=True, right=True, trim=True)
        b.set(xlabel="Tasks", ylabel=y_label)
        b.set_title(title)

        # step2: add notes for peaks
        boxes = b.patches
        for i, box in enumerate(boxes):
            if i < len(df)/len(boxes):    
                box_path = box.get_path()
                extents = box_path.get_extents()
                x, y, width, height = extents.xmin, extents.ymin, extents.width, extents.height
                # print(i, x, y, width, height)
                max_indices = df.groupby("difficulty")["reward"].idxmax()
                max_feature_value = df.loc[max_indices, "name"]
                max_value = df.loc[max_indices, "reward"]
                # set properties of annotations
                wid = 14 
                max_feature_value = textwrap.fill(max_feature_value.iloc[0], width=wid)
                fontdict={
                    'fontname': 'sans-serif', 
                    'fontsize': 8, 
                    'fontweight': 'bold'
                }

                b.text(
                    x + width/2, 
                    y + height + max_value.iloc[0]*0.05,  # translate by 8% of max value
                    max_feature_value,
                    ha='center', 
                    va='bottom', 
                    color='black', 
                    fontdict=fontdict
                )

        out_path = os.path.join(self.output_dir, f"{title}_boxplot.pdf")

        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    
    def normalization_analysis(self):
        # sub-step1: calculate standard value
        for data_dict_std in self.data:
            final_data = []
            std_dict = defaultdict(dict)
            system_name = f"{data_dict_std['agent_name']}_{data_dict_std['model_name']}_based_wo4"
            for env in ENVS:
                    for difficulty, data_lst in data_dict_std[env].items():
                        reward_lst = [json_obj["reward"] for json_obj in data_lst]
                        avg_reward = sum(reward_lst) / len(reward_lst)
                        std_dict[difficulty][env] = avg_reward
            # sub-step2: sequencely obtain normalized data
            for env in ENVS:
                for data_dict in self.data:
                    for difficulty, data_lst in data_dict[env].items():
                        reward_lst = [json_obj["reward"] for json_obj in data_lst]
                        avg_reward = sum(reward_lst) / len(reward_lst)
                        final_dict = {
                            "name": data_dict['model_name'],
                            "env": env,
                            "difficulty": difficulty,
                            "reward": avg_reward/std_dict[difficulty][env],
                        }
                        final_data.append(final_dict)
            df = pd.DataFrame(final_data)
            self.draw_boxplot(df, y_label="Normalized score", title=system_name)
            out_path = os.path.join(self.output_dir, f"{system_name}_normalized_data_wo4.jsonl")
            with open(out_path, 'w') as jsonl_file:
                jsonl_file.write(json.dumps(final_data, indent=2))
        print("Finished draw normalizational boxplot")
    

    def draw_lineplot(self, df, y_label, title):
        plt.rc('font',family='Times New Roman')
        sns.set_theme(style="whitegrid")
        l = sns.lineplot(
            data=df, x="comb_num", y="best_score", hue="line_type", palette="husl", 
            markers='so', style="env", linewidth=2,
        )
        sns.despine(offset=10, left=True)
        l.set(xlabel="Combined number", ylabel=y_label)
        l.set_title(title)
        l.set_xlim(1+0.5, len(self.data)+0.5)
        l.set_ylim(0, 1.1)
        l.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=2, fontsize=8)

        out_path = os.path.join(self.output_dir, f"{title}_lineplot.pdf")

        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


    def retrieval_analysis(self):
        pass

    def model_analysis(self):
        pass

    
    def combination_analysis(self):
        # # step1: draw performance histogram
        # best_system = self.data[0]
        # for data_dict in self.data:
        #     final_data = []
        #     system_name = f"{data_dict['agent_name']}_{data_dict['model_name']}"
        #     for env in ENVS:
        #         for difficulty, data_lst in data_dict[env].items():
        #             reward_lst = [json_obj["reward"] for json_obj in data_lst]
        #             avg_reward = sum(reward_lst) / len(reward_lst)
        #             final_dict = {
        #                 "env": env,
        #                 "difficulty": difficulty,
        #                 "reward": avg_reward,
        #             }
        #             final_data.append(final_dict)
        #     #         # step2: get best of n system for combination score
        #     #         for i, reward in enumerate(reward_lst):
        #     #             last_reward = best_system[env][difficulty][i]["reward"]
        #     #             if reward > last_reward:
        #     #                 best_system[env][difficulty][i] = data_lst[i]
        #     # df = pd.DataFrame(final_data)
        #     # self.draw_histogram(df, y_label="F1", title=system_name)
        # # step3: draw best system histogram
        # final_data = []
        # for env in ENVS:
        #     for difficulty, data_lst in best_system[env].items():
        #         reward_lst = [json_obj["reward"] for json_obj in data_lst]
        #         avg_reward = sum(reward_lst) / len(reward_lst)
        #         final_dict = {
        #             "env": env,
        #             "difficulty": difficulty,
        #             "reward": avg_reward,
        #         }
        #         final_data.append(final_dict)
        # df = pd.DataFrame(final_data)
        # self.draw_histogram(df, y_label="F1", title="best_system")
        # print("Finished draw performance histogram")

        

        # step5: deside best performance of model group on each task
        N = len(self.data)
        final_data = []
        for env in ENVS:
            for difficulty in DIFF:
                #sub-step1: traverse each combination num
                for i in range(2, N+1):
                    best_grp = ''
                    best_score = 1e8
                    grp = list(combinations(self.data, i))  # all possible combinations
                    len_tsk = len(grp[0][0][env][difficulty])
                    for g in grp:
                        name_comb = '+'.join([llm2abc[gg["model_name"]] for gg in g])
                        score_sum = 0
                        for j in range(0, len_tsk):  # sub-step2 traverse all instance
                            max_value = max(d[env][difficulty][j]['reward'] for d in g)
                            score_sum += max_value
                        score_sum = score_sum / len_tsk
                        if score_sum <= best_score:  # compare with best score
                            best_grp = name_comb
                            best_score = score_sum
                            print(f"get_best, task:{env},{difficulty}, name:{name_comb}, comb_num:{i}, score:{best_score:.4f}")
                    final_dict = {
                                "env": env,
                                "difficulty": difficulty,
                                "line_type": '_'.join((env,difficulty)),
                                "comb_num": i,
                                "best_group": best_grp,
                                "best_score": best_score
                            }
                    final_data.append(final_dict)
        
        df = pd.DataFrame(final_data)
        self.draw_lineplot(df, y_label="Best score", title="Best group for specific task-REV")
        #sub-ste keep processed data
        out_path = os.path.join(self.output_dir, f"Best_Group_origin_data_rev.jsonl")
        with open(out_path, 'w') as jsonl_file:
            jsonl_file.write(json.dumps(final_data, indent=2))
        print("Finished draw grouped lineplot")


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
    