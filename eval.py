import environment.wiki_run.utils as utils
import os
import json
agent_names = [ "React_wiki_run_Agent"]
model_names = ["llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"]
dataset_name = "hotpotqa"
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
for agent_name in agent_names:
    for model_name in model_names:
        max_length = model2maxlen[model_name]
        save_dir = f"data/noans_result/{agent_name}_{model_name}_{max_length}/"
        agent_save_file = os.path.join(save_dir, f"{dataset_name}_log.jsonl")
        if os.path.exists(agent_save_file):
            sessions = utils.get_all_agent_sessions(agent_save_file)
            completed_tasks = utils.get_non_error_tasks(sessions)
            print(f"{agent_name} with {model_name} finished {len(completed_tasks)} on {dataset_name}")
            recorded = []
            scores = []
            for session in sessions:
                if not session["error"]:
                    if session["question"] not in recorded:
                        scores.append(session["reward"])
                        recorded.append(session["question"])
            print(f"total recorded: {len(scores)}")
            print(f"average score: {sum(scores)/len(scores)}")
            print("-"* 16)
            
