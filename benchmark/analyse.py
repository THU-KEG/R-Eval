import argparse
import os
import pandas as pd
import seaborn as sns
import numpy as np
import random
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
        "KS": ["high_freq_ent", "low_freq_ent"],
        "KU": ["csj", "cpj", "cic"],
        "KA": ["hotpotqa", "2wikimultihopqa", "musique", "kqapro"]
    },
    "aminer" :{
        "KS": ["soay_easy"],
        "KU": ["profiling"],
        "KA": ["soay_hard"]
    }
}
LLM = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106",  "toolllama-2-7b", "llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b",  "codellama-13b-instruct"]
llm4dfsdt = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview", "toolllama-2-7b"]
llm4chatgptfunction = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
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

llm2short = {
    "llama2-7b-chat-4k": 'llama2-7b', 
    "tulu-7b": 'tulu-7b', 
    "llama2-13b": 'llama2-13b',
    "vicuna-13b": 'vicuna-13b', 
    "gpt-3.5-turbo-1106": 'gpt-3.5', 
    "gpt-4-1106-preview": 'gpt-4', 
    "codellama-13b-instruct": 'codellama-13b', 
    "toolllama-2-7b": 'toolllama2-7b'
}
llm2cost = {
    "llama2-7b-chat-4k": 7,
    "tulu-7b": 7,
    "llama2-13b": 13,
    "vicuna-13b": 13,
    "gpt-3.5-turbo-1106": 20,
    "gpt-4-1106-preview": 50,
    "codellama-13b-instruct": 13,
    "toolllama-2-7b": 7
}
# AGENT = ["React", "chatgpt_function", "PAL", "dfsdt"]
AGENT = ["React", "PAL", "dfsdt", "chatgpt_function"]
# AGENT = ["PAL"]
# AGENT = ["React"]
ENVS = ["wiki", "aminer"]
DIFF = ["KS", "KU", "KA"]

from matplotlib.colors import to_rgb

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

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
    parser.add_argument('--error_dir', type=str, default="data/error")
    parser.add_argument('--out_dir', type=str, default="data/analysis")
    # parser.add_argument('--environment', type=str, default="wiki", choices=ENVS)
    # agent args
    parser.add_argument('--model', type=str, default="tulu-7b", choices=LLM)
    parser.add_argument('--agent_name', type=str, default="React", choices=AGENT)
    parser.add_argument('--comp_model', type=str, default="vicuna-13b", choices=LLM)
    parser.add_argument('--comp_agent_name', type=str, default="React", choices=AGENT)
    # analyse args
    parser.add_argument('--aspect', type=str, default="normalization", choices=["performance", "error", "deploy", "combination", "normalization"])
    parser.add_argument('--function', type=str, default="all", choices=["single", "pairwise", "all", "error"])  
    return parser.parse_args(args)


class DataLoader:
    def __init__(self, result_dir, function, agent_name, model, comp_agent_name, comp_model, error_dir) -> None:
        self.result_dir = result_dir
        self.error_dir = error_dir
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
            if not os.path.exists(real_data_dir):
                print(f"{real_data_dir} not exists")
                continue
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

    def load_single_error_data(self, agent_name, model_name):
        data_dict = {
            "agent_name": agent_name,
            "model_name": model_name,
        }
        for env in ENVS:
            env_dict = {}
            dir_name = f"{agent_name}_{model_name}"
            real_data_dir = os.path.join(self.error_dir, dir_name)
            if not os.path.exists(real_data_dir):
                print(f"{real_data_dir} not exists")
                continue
            # under this dir there are many jsonl files, we load by difficulty
            datasets = env2datasets[env]
            for difficulty, dataset_list in datasets.items():
                data_lst = []
                for dataset in dataset_list:
                    jsonl_file = os.path.join(real_data_dir, f"{dataset}.json")
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            json_obj = json.loads(line)
                            data_lst.append(json_obj)
                # sort the data_lst by question
                env_dict[difficulty] = data_lst
            data_dict[env] = env_dict
        return data_dict
    
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
        elif self.function == "error":
            # get all error data
            data = []
            for agent_name in AGENT:
                model_list = LLM
                if agent_name == "dfsdt":
                    model_list = llm4dfsdt
                elif agent_name == "chatgpt_function":
                    model_list = llm4chatgptfunction
                for model_name in model_list:
                    data.append(self.load_single_error_data(agent_name, model_name))
        else:
            # get all data
            data = []
            for agent_name in AGENT:
                model_list = LLM
                if agent_name == "dfsdt":
                    model_list = llm4dfsdt
                elif agent_name == "chatgpt_function":
                    model_list = llm4chatgptfunction
                for model_name in model_list:
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
                                 args.model, args.comp_agent_name, args.comp_model, args.error_dir) 
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
        # delete rows with name "gpt-4-1106-preview"
        df = df[df.name != "gpt-4-1106-preview"]
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
                print(f"max_indices:{max_indices},\nmax_value:{max_value},\nmax_feature_value:{max_feature_value}\n")
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


    def draw_radar(self, data, value_list, agent):
        # 创建雷达图
        plt.rc('font',family='Times New Roman')
        plt.figure(figsize=(8, 8))
        sns.set_style("whitegrid")

        # 设置雷达图的角度和标签
        categories = list(data['Indicator'])
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # 绘制雷达图的轴线
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, fontsize=12)

        # 使用亮色绘制各个模型数据
        for values_name in value_list:
            values_series = list(data[values_name])
            values_series += values_series[:1]
            ax.plot(angles, values_series, 'o-', linewidth=1,markersize=3)
            ax.fill(angles, values_series, alpha=0.3, label=values_name)
        
        # 添加标题和图例
        # plt.title(title, fontsize=16)
        plt.legend(loc='best', title="Models",fontsize='xx-small')

        # 存储雷达图
        out_path = os.path.join(self.output_dir, f"{agent}_radar.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


    def performance_analysis(self):
        # calculate performance
        for agent in AGENT:
            final_dict = dict()
            final_data = []
            for data_dict in self.data:
                if data_dict['agent_name'] != agent:
                        continue
                for env in ENVS:
                    for difficulty, data_lst in data_dict[env].items():
                        reward_lst = [json_obj["reward"] for json_obj in data_lst]
                        avg_reward = sum(reward_lst) / len(reward_lst)
                        search_name = f'{data_dict["model_name"]}_{env}_{difficulty}'
                        final_dict[search_name] = avg_reward
                        dict_obj = {
                                "name": data_dict['model_name'],
                                "agent": data_dict['agent_name'],
                                "env": env,
                                "difficulty": difficulty,
                                "reward": avg_reward,
                            }
                        final_data.append(dict_obj)
            out_path = os.path.join(self.output_dir, f"{agent}_performance_data.jsonl")
            with open(out_path, 'w') as jsonl_file:
                jsonl_file.write(json.dumps(final_data, indent=2))
                
            # make dataframe
            indicator = []
            df_dict = {}
            for env in ENVS:
                for difficulty in DIFF:
                    indicator.append(f'{env}_{difficulty}')
            df_dict['Indicator'] = indicator
            
            model_list = LLM
            if agent == "dfsdt":
                model_list = llm4dfsdt
            elif agent == "chatgpt_function":
                model_list = llm4chatgptfunction
            for model in model_list:
                model_res = []
                for env in ENVS:
                    for difficulty in DIFF:
                        search_name = f'{model}_{env}_{difficulty}'
                        model_res.append(final_dict[search_name])
                df_dict[model] = model_res
            data = pd.DataFrame(df_dict)
            
            # draw and save graph
            self.draw_radar(data, model_list, agent)
        
    # def draw_pie(self, first_layer, second_layer, label_first, label_second, title):
    #     # plt.rc('font',family='Times New Roman')
    #     fig, ax = plt.subplots(figsize = (10, 10))
    #     size = 0.25
    #     font_size = 11
    #     # 获取colormap tab20c和tab20b的颜色
    #     cmap_c = plt.get_cmap("tab20c")
    #     cmap_b = plt.get_cmap("tab20b")

    #     # 使用tab20c的全部颜色和tab20b中的 5至8 颜色
    #     cmap_1 = cmap_c(np.arange(20))
    #     cmap_2 = cmap_b(np.array([4, 5, 6, 7]))

    #     # 内圈的颜色是每4个颜色中色彩最深的那个. vstack是将两类颜色叠加在一起
    #     inner_colors = np.vstack((cmap_1[::4], cmap_2[0]))
    #     # 外圈的颜色是全部24种颜色
    #     outer_colors = np.vstack((cmap_1, cmap_2))
        
    #     ax.pie(first_layer.flatten(), radius=1-size-size, 
    #             labels=label_first, 
    #             labeldistance=0.31,  rotatelabels=False, textprops={'fontsize': font_size}, 
    #             colors=inner_colors)
    #     ax.pie(second_layer.flatten(),   radius=1-size, colors=outer_colors,
    #             labels=label_second, 
    #             labeldistance=0.81,  rotatelabels=False, textprops={'fontsize': font_size}, 
    #             wedgeprops=dict(width=size, edgecolor='w'))
        
    #     out_path = os.path.join(self.output_dir, f"{title}_pie_graph.pdf")
    #     plt.savefig(out_path, bbox_inches='tight')
    #     plt.close()



    def draw_pie(self, first_layer, second_layer, label_first, label_second, title):
        fig, ax = plt.subplots(figsize = (10, 10))
        size = 0.25
        font_size = 11
        cmap_c = plt.get_cmap("tab20c")
        cmap_b = plt.get_cmap("tab20b")
        cmap_1 = cmap_c(np.arange(20))
        cmap_2 = cmap_b(np.array([4, 5, 6, 7]))
        inner_colors = np.vstack((cmap_1[::4], cmap_2[0]))
        outer_colors = np.vstack((cmap_1, cmap_2))
        
        # Lighten the colors
        inner_colors = [lighten_color(color, 0.5) for color in inner_colors]
        outer_colors = [lighten_color(color, 0.5) for color in outer_colors]

        ax.pie(first_layer.flatten(), radius=1-size-size, 
                labels=label_first, 
                labeldistance=0.31,  rotatelabels=False, textprops={'fontsize': font_size}, 
                colors=inner_colors)
        ax.pie(second_layer.flatten(),   radius=1-size, colors=outer_colors,
                labels=label_second, 
                labeldistance=0.81,  rotatelabels=False, textprops={'fontsize': font_size}, 
                wedgeprops=dict(width=size, edgecolor='w'))
        
        out_path = os.path.join(self.output_dir, f"{title}_pie_graph.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    def error_analysis(self):
        for agent in AGENT:
            model_list = LLM
            if agent == "dfsdt":
                model_list = llm4dfsdt
            elif agent == "chatgpt_function":
                model_list = llm4chatgptfunction
            for model in model_list:
                final_data = []
                for data_dict in self.data:
                    if data_dict['agent_name'] != agent or data_dict['model_name'] != model:
                            continue
                    for env in ENVS:
                        for difficulty, data_lst in data_dict[env].items():
                            ACC_lst = [json_obj["ACC"] for json_obj in data_lst]
                            avg_ACC = sum(ACC_lst) / len(ACC_lst)
                            WA_lst = [json_obj["WA"] for json_obj in data_lst]
                            avg_WA = sum(WA_lst) / len(WA_lst)
                            EM_lst = [json_obj["EM"] for json_obj in data_lst]
                            avg_EM = sum(EM_lst) / len(EM_lst)
                            AM_lst = [json_obj["AM"] for json_obj in data_lst]
                            avg_AM = sum(AM_lst) / len(AM_lst)
                            GE_lst = [json_obj["GE"] for json_obj in data_lst]
                            avg_GE = sum(GE_lst) / len(GE_lst)
                            ME_lst = [json_obj["ME"] for json_obj in data_lst]
                            avg_ME = sum(ME_lst) / len(ME_lst)
                            TE_lst = [json_obj["TE"] for json_obj in data_lst]
                            avg_TE = sum(TE_lst) / len(TE_lst)
                            RE_lst = [json_obj["RE"] for json_obj in data_lst]
                            avg_RE = sum(RE_lst) / len(RE_lst)
                            dict_obj = {
                                    "name": data_dict['model_name'],
                                    "agent": data_dict['agent_name'],
                                    "env": env,
                                    "difficulty": difficulty,
                                    "error_data": [
                                        [[avg_ACC], [avg_WA]],
                                        [[avg_EM, avg_AM], [avg_GE, avg_ME, avg_TE, avg_RE]]
                                    ]
                                }
                            final_data.append(dict_obj)
                out_path = os.path.join(self.output_dir, f'{agent}_{model}_error_analysis.jsonl')
                with open(out_path, 'w') as jsonl_file:
                    jsonl_file.write(json.dumps(final_data, indent=2))
                    
                # make dataframe
                first_layer_list = []
                for json_obj in final_data:
                    first_layer_list.append(np.array(json_obj["error_data"][0]))
                first_layer_data = sum(first_layer_list) / len(first_layer_list)
                second_layer_list = []
                for json_obj in final_data:
                    raw_data = json_obj["error_data"][1]
                    raw_data[0].append(0)
                    raw_data[0].append(0)
                    second_layer_list.append(np.array(raw_data))
                second_layer_data = sum(second_layer_list) / len(second_layer_list)
                
                first_layer_data = np.around(first_layer_data * 100, decimals = 1)
                second_layer_data = np.around(second_layer_data * 100, decimals = 1)
                
                first_label = [f"ACC\n{first_layer_data[0][0]}%", f"WA\n{first_layer_data[1][0]}%"]
                second_label = [f"EM\n{second_layer_data[0][0]}%",
                                f"AM\n{second_layer_data[0][1]}%",
                                "",
                                "",
                                
                                f"GE\n{second_layer_data[1][0]}%",
                                f"ME\n{second_layer_data[1][1]}%",
                                f"TE\n{second_layer_data[1][2]}%",
                                f"RE\n{second_layer_data[1][3]}%",
                                ]
                for i in range(8):
                    if second_layer_data.flatten()[i] == 0:
                        second_label[i] = ""

                self.draw_pie(first_layer_data, second_layer_data, first_label, second_label, f'{agent}_{model}')
    
    def draw_bubble(self, data, x_label, y_label, hue, size, title):
        plt.rc('font',family='Times New Roman')
        sns.set_theme(style="whitegrid")

        # Plot miles per gallon against horsepower with other semantics
        # g = sns.scatterplot(data=data, x='time', y='reward', size='cost', hue='model', sizes=(100, 2000), palette="husl")

        # Plot miles per gallon against horsepower with other semantics
        g = sns.scatterplot(data=data, x='time', y='reward', size='cost', hue='model', sizes=(100, 2000), palette="pastel", legend=False)
        
        # set x label
        g.set(xlabel="Execution time (s)")
        # set y label
        g.set(ylabel="F1 score")
        # Iterate over the rows of the data and add text
        for line in data.iterrows():
            if llm2short[line[1]['model']] in ["toolllama2-7b", "gpt-3.5", "vicuna-13b"]:
                g.text(line[1]['time'], line[1]['reward'], llm2short[line[1]['model']], horizontalalignment='center',
            verticalalignment='top', size='small')

            else:
                g.text(line[1]['time'], line[1]['reward'], llm2short[line[1]['model']], horizontalalignment='center',
            verticalalignment='bottom', size='small')


        out_path = os.path.join(self.output_dir, f"{title}_bubble_graph.pdf")

        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    
    def deploy_analysis(self):
        # calculate reward and time
        for domain in ENVS:
            final_data = []
            for data_dict in self.data:
                if data_dict["agent_name"] != "PAL":
                    continue
                env = domain
                
                
                # for difficulty, data_lst in data_dict[env].items():
                #     reward_lst = []
                #     time_lst = []
                #     for json_obj in data_lst:
                #         reward_lst.append(json_obj["reward"])
                #         time_lst.append(json_obj["exe_time"])
                #     avg_reward = sum(reward_lst) / len(reward_lst)
                #     if len(time_lst) > 100:
                #         time_lst = random.sample(time_lst, 100)
                #     avg_time = sum(time_lst) / len(time_lst)
                #     dict_obj = {
                #             "name": data_dict['model_name'],
                #             "env": env,
                #             "difficulty": difficulty,
                #             "reward": avg_reward,
                #             "exe_time": avg_time
                #         }
                #     final_data.append(dict_obj)
                
                reward_lst = []
                time_lst = []
                for difficulty, data_lst in data_dict[env].items():
                    for json_obj in data_lst:
                        reward_lst.append(json_obj["reward"])
                        time_lst.append(json_obj["exe_time"])
                avg_reward = sum(reward_lst) / len(reward_lst)
                if len(time_lst) > 100:
                    time_lst = random.sample(time_lst, 100)
                avg_time = sum(time_lst) / len(time_lst)
                dict_obj = {
                        "name": data_dict['model_name'],
                        "env": env,
                        "difficulty": difficulty,
                        "reward": avg_reward,
                        "exe_time": avg_time
                    }
                final_data.append(dict_obj)
                    
                    
            out_path = os.path.join(self.output_dir, f"{domain}_deploy_synth_data.jsonl")
            with open(out_path, 'w') as jsonl_file:
                jsonl_file.write(json.dumps(final_data, indent=2))
            
            # make dataframe
            model_cost = []
            reward_list = []
            exe_time_list = []
            model_name = []    
            for json_obj in final_data:
                model_name.append(json_obj["name"])
                exe_time_list.append(json_obj["exe_time"])
                reward_list.append(json_obj["reward"])
                model_cost.append(llm2cost[json_obj["name"]])
            data = pd.DataFrame({
                "model": model_name,
                "reward": reward_list,
                "time": exe_time_list,
                "cost": model_cost
            })
            
            # draw and save graph
            self.draw_bubble(data, "time", "reward", "model", "cost", f"{domain}_synth")
    
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
        if self.aspect == "performance":
            self.performance_analysis()
        elif self.aspect == "deploy":
            self.deploy_analysis()
        elif self.aspect == "normalization":
            self.normalization_analysis()
        elif self.aspect == "error":
            self.error_analysis()
        else:
            self.combination_analysis()


if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)
    analyzer = Analyzer(args)
    analyzer.run()
    