# KoLA2
KoLA2: Knowledge-oriented Language Agents Assessment


# How to run
You can use `pred.py` to run the agent tests. The usage is as follows:

```
nohup python pred.py --agent_name PlannerReact_wiki_run_Agent --llm_name gpt-3.5-turbo --max_context_len 4000 > PlannerReact_gpt-3.5-turbo_hotpot.nohup &
```

or just run the `shells/test_hotpotqa.sh` script.

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