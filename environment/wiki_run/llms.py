"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import requests
import json
import random
import tiktoken
token_enc = tiktoken.get_encoding("cl100k_base")
import openai
from langchain import PromptTemplate, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from environment.wiki_run.config import OPENAI_API_KEY, ONECHATS_API_KEY, ONECHAT4_API_KEY, SERVER_HOST, SERVER_PORT

OPENAI_CHAT_MODELS = ["gpt-3.5-turbo","gpt-3.5-turbo-16k-0613","gpt-3.5-turbo-16k","gpt-4-0613","gpt-4-32k-0613", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
OPENAI_LLM_MODELS = ["text-davinci-003","text-ada-001"]
FASTCHAT_MODELS = ["vicuna-7b"]
LLAMA_MODELS = ["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "llama2-13b", "vicuna-13b","codellama-13b-instruct", "toolllama-2-7b"]


class langchain_openai_chatllm:
    def __init__(self, llm_name):
        openai.api_key = OPENAI_API_KEY
        self.llm_name = llm_name
        human_template="{prompt}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
   
    def run(self, prompt, temperature=0, stop=['\n'], max_tokens=128):
        chat = ChatOpenAI(model=self.llm_name, temperature=temperature, stop=stop, max_tokens=max_tokens)
        self.chain = LLMChain(llm=chat, prompt=self.chat_prompt)
        return self.chain.run(prompt)

class agent_openai_chatllm:
    def __init__(self, llm_name):
        if "gpt-3.5" in llm_name:
            self.url = "https://sapi.onechat.fun/v1"
            self.api_key = ONECHATS_API_KEY
        else:
            self.url = "https://chatapi.onechat.fun/v1"
            self.api_key = ONECHAT4_API_KEY
        self.llm_name = llm_name
        human_template="{prompt}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    
    def run(self, prompt, temperature=0, stop=['\n'], max_tokens=128):
        chat = ChatOpenAI(
            openai_api_base=self.url, 
            openai_api_key=self.api_key,
            model=self.llm_name, temperature=temperature, stop=stop, max_tokens=max_tokens
            )
        self.chain = LLMChain(llm=chat, prompt=self.chat_prompt)
        # res = self.chain.run(prompt)
        try:
            res = self.chain.run(prompt)
        except TypeError:
            res = "Finish[no]"
        return res


class langchain_openai_llm:
    def __init__(self, llm_name):
        openai.api_key = OPENAI_API_KEY
        self.prompt_temp = PromptTemplate(
            input_variables=["prompt"], template="{prompt}"
        )
        self.llm_name = llm_name
        
    def run(self, prompt, temperature=0, stop=['\n'], max_tokens=128):
        llm = OpenAI(model=self.llm_name, temperature=temperature, stop=['\n'], max_tokens=max_tokens)
        chain = LLMChain(llm=llm, prompt=self.prompt_temp)
        return chain.run(prompt)


class langchain_fastchat_llm:
    def __init__(self, llm_name):
        openai.api_key = "EMPTY" # Not support yet
        openai.api_base = "http://localhost:8000/v1"
        self.prompt_temp = PromptTemplate(
            input_variables=["prompt"], template="{prompt}"
        )
        self.llm_name = llm_name
        
    def run(self, prompt, temperature=0.9, stop=['\n'], max_tokens=128):
        llm = OpenAI(model=self.llm_name, temperature=temperature, stop=['\n'], max_tokens=max_tokens)
        chain = LLMChain(llm=llm, prompt=self.prompt_temp)
        return chain.run(prompt)


class langchain_llama_llm:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.url = f"http://{SERVER_HOST}:{SERVER_PORT}/completion/{llm_name}"
        
    def run(self, prompt, temperature=1, stop=['\n'], max_tokens=128):
        
        data = {
            "input_text":prompt, 
            "max_new_tokens":max_tokens, 
            "stop_sequences": stop,
            "temperature":temperature,
            }
        result = requests.post(self.url, data=json.dumps(data))
        # return a string
        res = result.json()['completions_text']
        return res

def get_llm_backend(llm_name):
    if llm_name in OPENAI_CHAT_MODELS:
        # return langchain_openai_chatllm(llm_name)
        return agent_openai_chatllm(llm_name)
    elif llm_name in OPENAI_LLM_MODELS:
        return langchain_openai_llm(llm_name)
    elif llm_name in LLAMA_MODELS:
        return langchain_llama_llm(llm_name)
    else:
        return langchain_fastchat_llm(llm_name)
