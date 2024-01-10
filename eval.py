import environment.wiki_run.utils as utils
import os
import json

def main(dataset_name, agent_names, model_names):
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    for agent_name in agent_names:
        for model_name in model_names:
            max_length = model2maxlen[model_name]
            save_dir = f"data/result/{agent_name}_{model_name}_{max_length}/"
            agent_save_file = os.path.join(save_dir, f"{dataset_name}_log.jsonl")
            if os.path.exists(agent_save_file):
                sessions = utils.get_all_agent_sessions(agent_save_file)
                completed_tasks = utils.get_non_error_tasks(sessions)
                print(f"{agent_name} with {model_name} finished {len(completed_tasks)} on {dataset_name}")
                recorded = []
                scores = []
                has_error_num = 0 
                has_bad_pred_num = 0
                has_bad_score_num = 0
                for session in sessions:
                    error_str = "Could not find that page, please try again."
                    scratchpad = session["scratchpad"]
                    if error_str in scratchpad:
                        has_error_num += 1
                        if len(session["prediction"]) == 0:
                            has_bad_pred_num += 1
                        if session["reward"] == 0:
                            has_bad_score_num += 1
                    if not session["error"]:
                        if session["question"] not in recorded:
                            scores.append(session["reward"])
                            recorded.append(session["question"])
                        else:
                            scores.append(session["reward"])
                            print(f"Duplicate: {session['question']}")
                print(f"total recorded: {len(scores)}")
                if len(scores) == 0:
                    continue
                else:
                    print(f"average score: {sum(scores)/len(scores)}")
                print(f"has error: {has_error_num}")
                print(f"has bad score: {has_bad_score_num}")
                print(f"has bad pred: {has_bad_pred_num}")
                print("-"* 16)
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="high_freq_ent")
    parser.add_argument("--agent_names", type=str, nargs='+' , default=[ "React_wiki_run_Agent", "dfsdt_wiki_run_Agent","chatgpt_function_wiki_run_Agent", "PAL_wiki_run_Agent",   
    "React_aminer_run_Agent", "dfsdt_aminer_run_Agent", "chatgpt_function_aminer_run_Agent", "PAL_aminer_run_Agent"])
    parser.add_argument("--model_names", type=str, nargs='+', default=["llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"])
    args = parser.parse_args()
    main(args.dataset_name, args.agent_names, args.model_names)