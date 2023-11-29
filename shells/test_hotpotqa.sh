nohup python run_agent.py --agent_name Zeroshot_wiki_run_Agent --llm_name gpt-3.5-turbo --max_context_len 4000 > Zeroshot_gpt-3.5-turbo_hotpot.nohup &
nohup python run_agent.py --agent_name ZeroshotThink_wiki_run_Agent --llm_name gpt-3.5-turbo --max_context_len 4000 > ZeroshotThink_gpt-3.5-turbo_hotpot.nohup &
nohup python run_agent.py --agent_name React_wiki_run_Agent --llm_name gpt-3.5-turbo --max_context_len 4000 > React_gpt-3.5-turbo_hotpot.nohup &
nohup python run_agent.py --agent_name Planner_wiki_run_Agent --llm_name gpt-3.5-turbo --max_context_len 4000 > Planner_gpt-3.5-turbo_hotpot.nohup &
nohup python run_agent.py --agent_name PlannerReact_wiki_run_Agent --llm_name gpt-3.5-turbo --max_context_len 4000 > PlannerReact_gpt-3.5-turbo_hotpot.nohup &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-13b --environment wiki --dataset hotpotqa > data/logs/React_llama2-13b_hotpot.log &
nohup python3 pred.py --agent_name React_wiki_run_Agent --model vicuna-13b --environment wiki --dataset hotpotqa > data/logs/React_vicuna-13b_hotpot.log &