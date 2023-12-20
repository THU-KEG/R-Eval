"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import random
import re, string, os
import json 
import time
import requests
import tiktoken
from langchain.llms.base import BaseLLM
from collections import Counter

from environment.aminer_run.pre_prompt import react_agent_prompt
from environment.aminer_run.fewshots import REACT_EXAMPLE
import tiktoken
token_enc = tiktoken.get_encoding("cl100k_base")

def parse_action(string):
    pattern = r'^(\w+)\((.+)\)$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        action_type, argument = fuzzy_parse_action(string)
        return action_type, argument
        
def fuzzy_parse_action(text):
    text = text.strip(' ').strip('.')
    pattern = r'^(\w+)\((.+)\)'
    match = re.match(pattern, text)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return text, ''

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = token_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(token_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
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

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
  
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)

# def send_request(data, url):
#     headers = {'Content-Type': 'application/json'}
#     response = requests.post(url, data=json.dumps(data), headers=headers)
#     return response.json()

class AminerDocstore():
    def __init__(self):
        self.base_url = "http://soay.aminer.cn/"
        self.useless_person_info = ['interests', 'nameZh', 'avatar', 'activity']
        self.needed_pub_info = ['id', 'title' ,'year', 'authors', 'venue', 'pubAbstract', 'ncitation']
    
    # def searchPerson(self, name):
    #     query_data = {
    #         "query": name,
    #         "needDetails": True,
    #         "page": 0,
    #         "size": 10
    #     }
    #     url = self.base_url + "searchPerson"
    #     res1 = send_request(query_data, url)
    #     new_hitList = []
    #     for res in res1['data']['hitList']:
    #         new_data  = {}
    #         for key in res.keys():
    #             if key in self.useless_person_info:
    #                 continue
    #             new_data[key] = res[key]
    #         #     print(key, ": ", res[key])
    #         # print("================")
    #         # print("new_data: ", new_data)
    #         new_hitList.append(new_data)
    #     if new_hitList == []:
    #         raise "No person found"
    #     elif len(new_hitList) == 1:
    #         res_str = f"One person found: {json.dumps(new_hitList[0])}"
    #         return res_str
    #     else:
    #         res_str = f"{len(new_hitList)} people found: {json.dumps(new_hitList)}"
    #         return res_str

    # def searchPublication(self, name):
    #     query_data = {
    #         "query": name,
    #         "needDetails": True,
    #         "page": 0,
    #         "size": 10
    #     }
    #     url = self.base_url + "searchPublication"
    #     res1 = send_request(query_data, url)
    #     new_hitList = []
    #     for res in res1['data']['hitList']:
    #         new_data  = {}
    #         for key in res.keys():
    #             if key not in self.needed_pub_info:
    #                 continue
    #             new_data[key] = res[key]
    #         #     print(key, ": ", res[key])
    #         # print("================")
    #         # print("new_data: ", new_data)
    #         new_hitList.append(new_data)
    #     if new_hitList == []:
    #         raise "No publication found"
    #     elif len(new_hitList) == 1:
    #         res_str = f"One publication found: {json.dumps(new_hitList[0])}"
    #         return res_str
    #     else:
    #         res_str = f"{len(new_hitList)} publications found: {json.dumps(new_hitList)}"
    #         return res_str
    
    def searchPerson(self, **kwargs):
        personList = []
        addr = 'https://soay.aminer.cn/searchPerson'
        headers = {
            'Content-Type' : 'application/json'
        }
        searchKeyWordList = []
        if 'name' in kwargs:
            searchKeyWordList.append({
                        "operate": "0",
                        "wordType": 4,
                        "keyword": kwargs['name'],
                        "advanced": True,
                        "needTranslate": True
                    })
        if 'interest' in kwargs:
            searchKeyWordList.append({
                        "operate": "0",
                        "wordType": 2,
                        "keyword": kwargs['interest'],
                        "advanced": True,
                        "needTranslate": True
                    })
        if 'organization' in kwargs:
            searchKeyWordList.append({
                        "operate": "0",
                        "wordType": 5,
                        "keyword": kwargs['organization'],
                        "advanced": True,
                        "needTranslate": True
                    })
        json_content = json.dumps({
            "sort": [{'asc': False, 'field' : 'n_citation'}],
            "searchKeyWordList": searchKeyWordList,
            "needDetails" : True
        })
        response = requests.post(
            url=addr,
            headers = headers,
            data = json_content
        )
        result = response.json()
        for each in result['data']['hitList']:
            # print(each)
            try:
                new_data = {}
                for key in each.keys():
                    if key in self.useless_person_info:
                        continue
                    new_data[key] = each[key]
                new_data['interests'] = [each['interests'][i]['t'] for i in range(min(len(each['interests']), 10))]
                personList.append(
                    new_data
                )
            except:
                continue
        if personList == []:
            raise "No person found"
        elif len(personList) == 1:
            res_str = f"One person found: {json.dumps(personList[0])}"
            return res_str
        else:
            res_str = f"{len(personList)} people found: {json.dumps(personList)}"
            return res_str
    
    def searchPublication(self, publication_info):
        addr = 'https://soay.aminer.cn/searchPublication'
        pubList = []
        headers = {
            'Content-Type' : 'application/json'
        }
        json_content = json.dumps({
            "query" : publication_info,
            'needDetails' : True,
            'page' : 0,
            'size' : 10,
            "sort": [{'asc': False, 'field' : 'n_citation'}],
        })
        response = requests.post(
            url=addr,
            headers = headers,
            data = json_content
        )
        result = response.json()
        new_hitList = []
        for res in result['data']['hitList']:
            new_data  = {}
            for key in res.keys():
                if key not in self.needed_pub_info:
                    continue
                new_data[key] = res[key]
            #     print(key, ": ", res[key])
            # print("================")
            # print("new_data: ", new_data)
            new_hitList.append(new_data)

        if new_hitList == []:
            raise "No publication found"
        elif len(new_hitList) == 1:
            res_str = f"One publication found: {json.dumps(new_hitList[0])}"
            return res_str
        else:
            res_str = f"{len(new_hitList)} publications found: {json.dumps(new_hitList)}"
            return res_str

    def getPersonPubs(self, id):
        url = 'https://soay.aminer.cn/getPersonPubs'
        addr = f"{url}?id={id}&offset=0&size=10&order=citation"
        response = requests.get(url=addr)
        result = response.json()['data'][0]['data']['pubs']
        pub_list = []
        for each in result:
            try:
                pub_list.append({
                    # 'abstract' : result[i]['abstract'],
                    'pub_id' : each['id'],
                    'title' : each['title'],
                    'num_citation' : each['ncitation'],
                    'year' : each['year'],
                    'authors_name_list' : [each['authors'][j]['name']for j in range(len(each['authors']))]
                })
            except:
                continue
        return json.dumps(pub_list)

    def getCoauthors(self, id):
        url = 'https://soay.aminer.cn/getCoauthors'
        addr = f"{url}?id={id}"
        response = requests.get(url=addr)
        # print("=====================================")
        # print(response.json())
        # print("==============")
        # print(response.json()['data'][0]['data'])
        # print("=====================================")
        result = response.json()['data'][0]['data']['crs']
        coauthorsList = []
        for each in result:
            try:
                coauthorsList.append({
                    'person_id' : each['id'],
                    'name' : each['name'],
                    'relation' : each['relation']
                })
            except:
                continue
        # coauthorsList = [{'person_id' : result[i]['id'], 'relation' : result[i]['relation']} for i in range(min(len(result), 10))]
        coauthorsList_str =  json.dumps(coauthorsList)
        return coauthorsList_str


class BaseAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 llm: BaseLLM,
                 context_len: int = 2000,
                 max_steps: int= 10,
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = ""
        self.examples = ""
        self.context_len = context_len
        self.run_error = False
        self.name = "Base_aminer_run_Agent"
        self.docstore = AminerDocstore()
        self.llm = llm
        
        self.enc = token_enc
        self.__reset_agent()
    
    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished() and not self.run_error:
            self.step()
    
    def prompt_agent(self) -> str:
        generation = self.llm(self._build_agent_prompt())
        self.check_run_error(generation)
        return format_step(generation)
 
    def check_run_error(self, text):
        if text in ["No response"]:
            self.run_error = True
            
    def is_finished(self) -> bool:
        return self.finished
    
    def reward(self) -> float:
        return f1_score(self.answer, self.key)   
    
    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps)
                or (len(self.enc.encode(self._build_agent_prompt())) > self.context_len)
                ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

    def _think(self):
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])
    
    def _action(self):
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        return action_type, argument
        
    def step(self) -> None:
        
        # agent forward
        ret = self.forward()
        if ret:
            action_type, argument = ret[0], ret[1]
        else:
            action_type = ret
        
        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'searchPerson':
            try:
                kwargs = eval("{" + 
                              argument.replace("name", "'name'").replace("interest", "'interest'").replace("organization", "'organization'").replace("=", ": ") 
                              + "}")
                self.scratchpad += format_step(self.docstore.searchPerson(**kwargs))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that person, please try again.'
        
        elif action_type == 'searchPublication':
            try:
                self.scratchpad += format_step(self.docstore.searchPublication(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that publication, please try again.'
        
        elif action_type == 'getCoauthors':
            try:
                # print("getCoauthors argument is")
                print(len(argument))
                # print(argument)
                argument_str = argument.strip('\'').strip('\"')
                self.scratchpad += format_step(self.docstore.getCoauthors(argument_str))
            except ValueError:
                self.scratchpad += f'The id may not be a person\'s id for using getCoauthors. Please try other actions.'

        elif action_type == 'getPersonPubs':
            try:
                argument_str = argument.strip('\'').strip('\"')
                self.scratchpad += format_step(self.docstore.getPersonPubs(argument_str))
            except ValueError:
                self.scratchpad += f'The id may not be a person\'s id for using getPersonPubs.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are searchPerson(name=<name>, organization=<organization>, interest=<interest>), searchPublication(<name>), getCoauthors(<id>), getPersonPubs(<id>) and Finish(<answer>). Please try other actions.'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1
    
    def _build_agent_prompt(self) -> str:
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    
class ReactAgent(BaseAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 context_len: int = 2000
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = REACT_EXAMPLE
        self.agent_prompt = react_agent_prompt
        self.name = "React_aminer_run_Agent"
    
    def forward(self):
        self._think()
        action_type, argument = self._action()
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
        



def get_aminer_agent(agent_name, dataset_name):
    if agent_name in ["React_aminer_run_Agent"]:
        return ReactAgent
