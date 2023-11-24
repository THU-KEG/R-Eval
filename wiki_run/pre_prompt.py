"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

from langchain.prompts import PromptTemplate

ZEROSHOT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search. For example, Search[Milhouse]
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search. For example, Lookup[named after]
(3) Finish[answer], which returns the answer and finishes the task. For example, Finish[Richard Nixon] 
You may take as many steps as necessary.
Question: {question}{scratchpad}"""

zeroshot_agent_prompt = PromptTemplate(
                        input_variables=["question", "scratchpad"],
                        template = ZEROSHOT_INSTRUCTION,
                        )

REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REACT_INSTRUCTION,
                        )

PLAN_INSTRUCTION = """Setup a plan for answering question with Actions. Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
{examples}
(END OF EXAMPLES)
Question: {question}
Plan:"""

plan_prompt = PromptTemplate(
                input_variables=["examples", "question"],
                template = PLAN_INSTRUCTION,
            )

PLANNER_INSTRUCTION = """Solve a question answering task with Plan, interleaving Action, Observation steps. Plan is decided ahead of Actions. Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}
Plan: {plan}{scratchpad}"""

planner_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "plan", "scratchpad"],
                        template = PLANNER_INSTRUCTION,
                        )

PLANNERREACT_INSTRUCTION = """Solve a question answering task with Plan, interleaving Thought, Action, Observation steps. Plan is decided ahead of Actions. Thought can reason about the current situation. Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}
Plan: {plan}{scratchpad}"""

plannerreact_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "plan", "scratchpad"],
                        template = PLANNERREACT_INSTRUCTION,
                        )


