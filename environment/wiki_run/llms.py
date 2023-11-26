"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import sys
import json
import random
import tiktoken
token_enc = tiktoken.get_encoding("cl100k_base")
import openai
from langchain import PromptTemplate, OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from environment.wiki_run.config import OPENAI_API_KEY

OPENAI_CHAT_MODELS = ["gpt-3.5-turbo","gpt-3.5-turbo-16k-0613","gpt-3.5-turbo-16k","gpt-4-0613","gpt-4-32k-0613", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
OPENAI_LLM_MODELS = ["text-davinci-003","text-ada-001"]
FASTCHAT_MODELS = ["vicuna-7b"]
LLAMA_MODELS = ["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "internlm-7b-8k"]


class langchain_openai_chatllm:
    def __init__(self, llm_name):
        openai.api_key = OPENAI_API_KEY
        self.llm_name = llm_name
        human_template="{prompt}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
   
    def run(self, prompt, temperature=1, stop=['\n'], max_tokens=128):
        chat = ChatOpenAI(model=self.llm_name, temperature=temperature, stop=stop, max_tokens=max_tokens)
        self.chain = LLMChain(llm=chat, prompt=self.chat_prompt)
        return self.chain.run(prompt)
        
class langchain_openai_llm:
    def __init__(self, llm_name):
        openai.api_key = OPENAI_API_KEY
        self.prompt_temp = PromptTemplate(
            input_variables=["prompt"], template="{prompt}"
        )
        self.llm_name = llm_name
        
    def run(self, prompt, temperature=0.9, stop=['\n'], max_tokens=128):
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

def get_llm_backend(llm_name):
    if llm_name in OPENAI_CHAT_MODELS:
        return langchain_openai_chatllm(llm_name)
    elif llm_name in OPENAI_LLM_MODELS:
        return langchain_openai_llm(llm_name)
    else:
        return langchain_fastchat_llm(llm_name)
