import os
import argparse 
import numpy as np
import pandas as pd
import concurrent
import joblib
from environment.wiki_run.utils import summarize_trial_detailed, log_trial
import environment.wiki_run.utils as utils
from environment.wiki_run.agent_arch import get_agent
from environment.wiki_run.llms import get_llm_backend
from environment.wiki_run.config import available_agent_names


parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="React")
parser.add_argument("--llm_name", type=str, help="Name of the llm", default="gpt-3.5-turbo")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=1700)
args = parser.parse_args()

agent_name = args.agent_name
llm_name = args.llm_name
max_context_len = args.max_context_len
assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()

def run_one_complex_level(level="easy"):
    hotpot = joblib.load(f'data/raw/hotpotqa/{level}.joblib').reset_index(drop = True)
    agent_save_file = f"data/result/hotpotqa/{level}_{agent_name}_{llm_name}.jsonl"
    task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        print(f"{level}:{len(completed_tasks)}")
        task_instructions = [task for task in task_instructions if task not in completed_tasks]
        utils.delete_error(agent_save_file)
    llm = get_llm_backend(llm_name).run
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, ans, llm, max_context_len) for ques, ans in task_instructions]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_agent_run_step, agents)
    for agent in agents:
        utils.log_agent(agent, agent_save_file)
    print(f'Finished Trial. Total: {len(agents)}')
    
def main():
    levels = ['easy', 'medium', 'hard']
    for level in levels:
        run_one_complex_level(level)
    
if __name__ == '__main__':
    main()