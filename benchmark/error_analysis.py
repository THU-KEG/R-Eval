import re
from collections import Counter
import string
from benchmark.analyse import ENVS, model2maxlen, env2datasets
import jsonlines
import json
from tqdm import tqdm
import os
AGENT = ["React", "chatgpt_function", "PAL", "dfsdt"]

class analyser():
    def __init__(self, f1_threshold) -> None:
        self.query_quantity = 100
        self.f1_threshold = f1_threshold
        self.model_list = ['llama2-7b-chat-4k', 'tulu-7b', 'llama2-13b', 'vicuna-13b', 'codellama-13b-instruct', 'toolllama-2-7b', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview']
        self.task_list = ['1-3_soay_easy', '2-4_profiling', '3-5_soay_hard']

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        
        def white_space_fix(text):
            return " ".join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def calculate_error(self, result_file):
        EM = AM = GE = ME = TE = RE = 0
        with jsonlines.open(result_file, 'r') as f:
            for data in f:
                if data['reward'] >= self.f1_threshold:
                    if self.normalize_answer(str(data['answer'])) in self.normalize_answer(str(data['scratchpad'])):
                    # if self.f1_score(prediction = str(data['scratchpad']), ground_truth = str(data['answer']))[0] >= 0.01:
                        EM += 1
                    else:
                        AM += 1
                else:
                    if self.normalize_answer(str(data['answer'])) in self.normalize_answer(str(data['scratchpad'])):
                    # if self.f1_score(prediction = str(data['scratchpad']), ground_truth = str(data['answer']))[0] >= 0.01:
                        GE += 1
                    else:
                        if data['halted']:
                            if data['error']:
                                ME += 1
                            else:
                                TE += 1
                        else:
                            RE += 1

            ACC = EM + AM
            WA = GE + ME + TE + RE

            result_doc = {
                'ACC' : ACC / 100,
                'WA' : WA / 100,
                'EM' : EM / 100,
                'AM' : AM / 100,
                'GE' : GE / 100,
                'ME' : ME / 100,
                'TE' : TE / 100,
                'RE' : RE / 100
            }

            f.close()
        return result_doc


    def error_analysis_all(self):
        # model_bar = tqdm(self.model_list, desc = 'processing of models')
        result_dir = "data/result"
        for agent_name in AGENT:
            for model_name in self.model_list:
                for env in ENVS:
                    max_length = model2maxlen[model_name]
                    dir_name = f"{agent_name}_{env}_run_Agent_{model_name}_{max_length}"
                    real_dir = os.path.join(result_dir, dir_name)
                    if not os.path.exists(real_dir):
                        continue
                    output_dir = './data/error/{}_{}'.format(agent_name, model_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # load
                    datasets = env2datasets[env]
                    for diff, tasks in datasets.items():
                        for task in tasks:
                            result_file = os.path.join(real_dir, '{}_log.jsonl'.format(task))
                            if not os.path.exists(result_file):
                                continue
                            result_doc = self.calculate_error(result_file)
                            # dump
                            output_file =  os.path.join(output_dir, '{}.json'.format(task))
                            print('model: {}, task: {}, result_doc: {}'.format(model_name, task, result_doc))
                            with open(output_file, 'w') as f:
                                json.dump(result_doc, f)
                                f.close()


if __name__ == '__main__':
    sys = analyser(f1_threshold=0.05)
    sys.error_analysis_all()