import environment.wiki_run.utils as utils
import os
import json
import datetime

def remove_save(file_name, sessions, non_error_sessions):
    # get date str
    x = datetime.datetime.now()
    with open(file_name+'.back', 'a') as b_f:
        b_f.write(f"backup at {x}\n")
        for sess in sessions:
            json.dump(sess, b_f)
            b_f.write('\n')
    with open(file_name, 'w') as f:
        for sess in non_error_sessions:
            json.dump(sess, f)
            f.write('\n')

def main(dataset_name, agent_names, model_names, remove_state):
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
                has_error_num = 0 
                has_bad_pred_num = 0
                has_bad_score_num = 0
                no_error_sessions = []
                not_bad_pred = []
                not_bad_score = []
                for session in sessions:
                    error_str = "Could not find that page, please try again."
                    scratchpad = session["scratchpad"]
                    # for remove
                    if error_str in scratchpad:
                        has_error_num += 1
                        if len(session["prediction"]) == 0:
                            has_bad_pred_num += 1
                        else:
                            not_bad_pred.append(session)
                        if session["reward"] == 0:
                            has_bad_score_num += 1
                        else:
                            not_bad_score.append(session)
                    else:
                        no_error_sessions.append(session)
                # remove error tasks
                if remove_state == 'bad_score':
                    remove_save(agent_save_file, sessions, not_bad_score + no_error_sessions)
                elif remove_state == 'bad_pred':
                    remove_save(agent_save_file, sessions, not_bad_pred + no_error_sessions)    
                print(f"has error: {has_error_num}")
                print(f"has bad score: {has_bad_score_num}")
                print(f"has bad pred: {has_bad_pred_num}")
                print("-"* 16)
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove_state", type=str, default="bad_score", choices=['bad_score', 'bad_pred'])
    parser.add_argument("--dataset_name", type=str, default="high_freq_ent")
    parser.add_argument("--agent_names", type=str, nargs='+' , default=[ "React_wiki_run_Agent"])
    parser.add_argument("--model_names", type=str, nargs='+', default=["llama2-7b-chat-4k", "tulu-7b", "llama2-13b", "vicuna-13b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "codellama-13b-instruct", "toolllama-2-7b"])
    args = parser.parse_args()
    main(args.dataset_name, args.agent_names, args.model_names, args.remove_state)