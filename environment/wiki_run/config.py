"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""
from config.server import openai_api_key, onechat_s_key, onechat_4_key, server_host, server_port
available_agent_names = ["Zeroshot_wiki_run_Agent","ZeroshotThink_wiki_run_Agent","React_wiki_run_Agent","Planner_wiki_run_Agent","PlannerReact_wiki_run_Agent"]
OPENAI_API_KEY = openai_api_key
ONECHATS_API_KEY = onechat_s_key
ONECHAT4_API_KEY = onechat_4_key
SERVER_HOST = server_host
SERVER_PORT = server_port