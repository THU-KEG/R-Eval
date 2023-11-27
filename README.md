# KoLA2
KoLA2: Knowledge-oriented Language Agents Assessment

# Requirements

For client:
```
pandas
gym
numpy
tiktoken
langchain
wikipedia
```

For gpu server:
```
transformers
torch
```

# How to run
You can use `pred.py` to run the agent tests. The usage is as follows:

```
nohup python pred.py --agent_name React_wiki_run_Agent --model llama2-7b-chat-4k --environment wiki --dataset hotpotqa > data/logs/React_llama2-7b-chat_hotpot.log &
```

or just run the `shells/test_hotpotqa.sh` script.

# For GPU Server
Apart from those OpanAI models which can be called via API, we also need some open-source models to run like `llama2` and `tulu`. 

On the gpu server，we use FastAPI to run the models. First enter the `environment/server` folder, then run the command as follows (**Please set the model_names in infer.py first**):

```
CUDA_VISIBLE_DEVICES=2,3 nohup uvicorn infer:app --host '0.0.0.0' --port 9627  > models.log &
```

Note that the number of model_names should be equal to the number of GPUs. For example, if you have 2 models in the model_names, then you should have 2 GPUs.

# For Wikipedia Environment
We directly use the LangChain's DocstoreExplorer which can search the Wikipedia articles by the title.
The file at wiki_run/wikienv.py is similar to the files in LangChain, but wikienv.py is actually not used.

Here are the usage of the files in wiki_run:
- For agent:
    - agent_arch.py: the agent architectures, most of the running logics are in this file
    - fewshot.py: the examples of few-shot learning
    - pre_prompt.py: the prompts for wikipedia QA tasks
    - config.py: the configuration of apis
    - llms.py: the language models
- For environment:
    - wikienv.py: the environment for Wikipedia
    - wrappers.py: the wrappers for the environment
    - utils.py: the utils for the environment
    - evaluate.py: the evaluation of the environment

# For AMiner Environment

TODO