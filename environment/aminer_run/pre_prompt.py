"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

from langchain.prompts import PromptTemplate

REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be five types: 
(1) searchPerson(name=<name>, organization=<organization>, interest=<interest>), which searches the person by his or her <name> on Aminer, an academic scholar database (besides <name>,the <organization> and <interest> are also two optional parameters) and returns the information of the people found.
(2) searchPublication(<name>), which searches the publication by <name> on  Aminer and returns the top 10 publications found.
(3) getCoauthors(<id>), which returns a person's coauthors using the person's id found by searchPerson.
(4) getPersonPubs(<id>), which returns a person's publications using the person's id found by searchPerson.
(5) Finish(<answer>), which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REACT_INSTRUCTION,
                        )
