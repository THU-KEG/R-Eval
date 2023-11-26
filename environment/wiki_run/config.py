"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""
from config.server import openai_api_key, server_host
available_agent_names = ["Zeroshot_wiki_run_Agent","ZeroshotThink_wiki_run_Agent","React_wiki_run_Agent","Planner_wiki_run_Agent","PlannerReact_wiki_run_Agent"]
OPENAI_API_KEY = openai_api_key
SERVER_HOST = server_host
MODEL2PORT = {
    "llama2-7b-chat-4k": "9626",
    "tulu-7b": "9627"
}