# R-Eval
R-Eval: A Unified Toolkit for Evaluating Domain Knowledge of Retrieval Augmented Large Language Models
(In submission)

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
python3 pred.py --agent_name React_wiki_run_Agent --model vicuna-13b --environment wiki --dataset hotpotqa --num_workers 1
```

or just run the `shells/test_hotpotqa.sh` script.

# For GPU Server
Apart from those OpanAI models which can be called via API, we also need some open-source models to run like `llama2` and `tulu`. 

On the gpu server，we use FastAPI to run the models. First enter the `environment/server` folder, then run the command as follows (**Please set the model_names in infer.py first**):

```
CUDA_VISIBLE_DEVICES=5,7 nohup uvicorn infer:app --host '0.0.0.0' --port 9627  > models.log &
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

The files are in aminer_run, we provide these apis for the AMiner environment:

- **searchPerson**. The searchPerson function, which is based on the scholar entities' information in Aminer, receives the name, organization and interest of this intended scholar and returns the detailed information including person's id, citation number and publication number  via fuzzy match.


- **searchPublication**. The searchPublication function, which is based on the publication entities' information in Aminer, receives the publication information and returns the related information including publication's id, title and publication year via fuzzy match.


- **getCoauthors**. The getCoauthors function, which is based on the relation information between scholar entities, receives the person id then returns the input scholar's coauthors and their detailed information including id, name and relation via exact match.

- **getPersonInterest**. The getPersonInterest function, which is based on the property information of scholar entities in Aminer, receives the scholar's id and returns a list of the person's interested research topics via exact match.

- **getPublication**. The getPublication function, which is based on the property information of publication entities in Aminer, receives the publication's id and returns its detailed information including the publication's abstract, author list and the number of citation via exact match.

- **getPersonBasicInfo**. The getPersonBasicInfo function, which is based on the scholar entities' property information in Aminer, receives the person's id of this intended scholar and returns the detailed information including person's name, gender, organization, position, short bio, education experience and email address via exact match. In fact, these information consists of the person's profile.

- **getPersonPubs**. The getPersonPubs function, which is based on the relation information between publication entities and  scholar entities in Aminer, receives the person's id, and returns the detailed information including the publication's id, title, citation number and the authors' name list via exact match.
