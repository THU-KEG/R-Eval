from benchmark.analyse import DataLoader, parse_args, ENVS, model2maxlen, dataset2level, env2datasets
import os
import pandas as pd
import json
datasets = ["high_freq_ent",
    "low_freq_ent",
    "csj",
    "cpj",
    "cic",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "kqapro","soay_easy",
    "profiling",
    "soay_hard"]
level_wiki = ["high_freq_ent",
    "low_freq_ent",
    "csj",
    "cpj",
    "cic",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "kqapro"]
level_aminer = ["soay_easy",
    "profiling",
    "soay_hard"]
agent_name2short = {
    "React": "ReAct",
    "PAL": "PAL",
    "dfsdt": "DFSDT",
    "chatgpt_function": "FC",
}

model_name2short = {
    "llama2-7b-chat-4k": "llama2-7b-chat",
    "tulu-7b": "tulu-7b",
    "llama2-13b": "llama2-13b",
    "vicuna-13b": "vicuna-13b",
    "codellama-13b-instruct": "codellama-13b",
    "toolllama-2-7b": "toolllama2-7b",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
    "gpt-4-1106-preview": "gpt-4-1106",
}

def gen_table(args):
    data_loader = DataLoader(args.result_dir, "all", args.agent_name, 
                                 args.model, args.comp_agent_name, args.comp_model, args.error_dir)
    data = data_loader.load_data()
    print("load models num: ", len(data))
    # calculate the average reward of each model on each dataset
    model_avg_score_list = []
    for data_dict in data:
        # print(data_dict.keys())
        agent_name = data_dict["agent_name"]
        model_name = data_dict["model_name"]
        new_dict = {
            "Arch.": agent_name2short[agent_name],
            "LLM": model_name2short[model_name],
        }
        env2scores = {}
        for env in ENVS:
            env_scores = []
            max_length = model2maxlen[model_name]
            dir_name = f"{agent_name}_{env}_run_Agent_{model_name}_{max_length}"
            real_data_dir = os.path.join(args.result_dir, dir_name)
            if not os.path.exists(real_data_dir):
                print(f"{real_data_dir} not exists")
                continue
            # under this dir there are many jsonl files, we load by difficulty
            datasets = env2datasets[env]
            for difficulty, dataset_list in datasets.items():
                for dataset in dataset_list:
                    jsonl_file = os.path.join(real_data_dir, f"{dataset}_log.jsonl")
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        task_scores = []
                        for line in f:
                            json_obj = json.loads(line)
                            reward = json_obj["reward"]
                            env_scores.append(reward)
                            task_scores.append(reward)
                        _avg = sum(task_scores) / len(task_scores) * 100
                        new_dict[dataset] = _avg
            env2scores[env] = env_scores
        # for env, env_scores in env2scores.items():
        #     env_avg = sum(env_scores) / len(env_scores) * 100
        #     new_dict[f"avg_{env}"] = env_avg
        model_avg_score_list.append(new_dict)
    # generate latex table, each row is a model's performance on a dataset
    # each column is a dataset's performance
    df = pd.DataFrame(model_avg_score_list)
    print(df)
    create_standard_table(level_wiki, model_avg_score_list, 2, 3)
    create_standard_table(level_aminer, model_avg_score_list, 1, 1)
    

def number_to_rank(number):
    if number == 1:
        return '1st'
    elif number == 2:
        return '2nd'
    elif number == 3:
        return '3rd'
    else:
        return '%dth' % number

def get_true_rank(i, rank_with_id):
    for rank_id in rank_with_id:
        if rank_id[0] == i:
            return rank_id[1]


def get_color(row_id, data_level='5'):
    row_id = int(row_id)
    data_level = str(data_level)
    # print(row_id)
    if len(data_level) > 1:
        data_level = data_level[0]

    color_name = ''
    if row_id % 2 == 0:
        color_name += 'deep'
    else:
        color_name += 'shallow'
    color_name += data_level
    return color_name

def create_standard_table(levels, dataset2final_scores, level1_len, level2_len):
    # final_scores are freezed
    df = pd.DataFrame(dataset2final_scores)
    df.to_csv('data/dataset_final_scores.csv')
    # sumarize df by row
    data_df = df[datasets]
    sum_df = data_df.sum(axis=1) / len(data_df.columns)
    # rank by sum_df
    total_rank = sum_df.rank(method='average', ascending=False)
    print("total_rank")
    print(total_rank)
    rank_with_id = [(i, rank) for i, rank in enumerate(total_rank)]
    rank_with_id.sort(key=lambda x: x[1], reverse=False)
    print("rank_with_id", rank_with_id)
    # 所有任务上的分数
    if levels == level_wiki:
        local_df = data_df.iloc[:, :9]
    else:
        local_df = data_df.iloc[:, 9:]
    # get wiki and aminer rank
    # print("df[level_wiki]\n", df[level_wiki])
    # print("df[level_aminer]\n", df[level_aminer])
    wiki_scores = df[level_wiki].sum(axis=1) / len(level_wiki)
    aminer_scores = df[level_aminer].sum(axis=1) / len(level_aminer)
    wiki_rank = df[level_wiki].sum(axis=1).rank(
        method='average', ascending=False)
    aminer_rank = df[level_aminer].sum(axis=1).rank(
        method='average', ascending=False)
    # get level ranks
    l1_cols = local_df.columns[0:level1_len]
    l2_cols = local_df.columns[level1_len:level1_len + level2_len]
    l3_cols = local_df.columns[level1_len + level2_len:]
    l1_ranks = local_df[l1_cols].sum(axis=1).rank(
        method='average', ascending=False)
    l2_ranks = local_df[l2_cols].sum(axis=1).rank(
        method='average', ascending=False)
    l3_ranks = local_df[l3_cols].sum(axis=1).rank(
        method='average', ascending=False)

    # get table rows
    system2row = {}
    for i, rank in enumerate(rank_with_id):
        true_rank = get_true_rank(i, rank_with_id)
        data_row = df.iloc[i,:]
        # print(data_row)
        true_agent_name = data_row["Arch."]
        true_llm_name = data_row["LLM"]
        if int(true_rank) % 2 == 0:
            model_color_name = 'gry'

        else:
            model_color_name = 'wit'
        model_row_str = f"\cellcolor{ {model_color_name} } {true_agent_name} & \cellcolor{ {model_color_name} } {true_llm_name} & "
        true_color = 1
        for j, dataset_name in enumerate(levels):
            final_score = data_row[dataset_name]
            # 如果 是第二个表的最后一系统，则打印一些其他的
            model_score = round(final_score, 1)
            model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} } {model_score} & "
            
            # 结尾加上rank
            if j == level1_len - 1:
                model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {number_to_rank(l1_ranks[i])} & "
                true_color = 2
            elif j == level1_len + level2_len - 1:  
                model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {number_to_rank(l2_ranks[i])} & "
                true_color = 3
            elif j == len(levels) - 1:
                model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {number_to_rank(l3_ranks[i])} & "
            
        # 如果 是第二个表的最后一系统，则打印一些其他的 wiki_rank, aminer_rank, total_rank
        if dataset_name == 'soay_hard':
            true_color = 4
            wiki_score = round(wiki_scores[i], 1)
            aminer_score = round(aminer_scores[i], 1)
            overall_score = round(sum_df[i], 1) 
            model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {wiki_score} & "
            model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {number_to_rank(wiki_rank[i])} & "
            model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {aminer_score} & "
            model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {number_to_rank(aminer_rank[i])} & "
            model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {overall_score} & "
            model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {number_to_rank(total_rank[i])} & "
        # else:
        # 否则就加上 \\\\  
        model_row_str = model_row_str[:-2]
        model_row_str += f" \\\\ "
        system2row[i] = model_row_str

    # output table in the rank order
    print('-------------------')
    for i, rank in rank_with_id:
        print(system2row[i])
    print('-------------------')


if __name__ == '__main__':
    args = parse_args()
    gen_table(args)

