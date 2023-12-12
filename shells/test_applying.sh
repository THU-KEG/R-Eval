nohup python3 pred.py --agent_name React_wiki_run_Agent --model vicuna-13b --environment wiki --dataset kqapro >> data/logs/React/vicuna-13b/kqapro.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-13b --environment wiki --dataset kqapro >> data/logs/React/llama2-13b/kqapro.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model tulu-7b --environment wiki --dataset kqapro >> data/logs/React/tulu-7b/kqapro.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-7b-chat-4k --environment wiki --dataset kqapro >> data/logs/React/llama2-7b-chat-4k/kqapro.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model codellama-13b-instruct --environment wiki --dataset kqapro --num_workers 1 >> data/logs/React/codellama-13b-instruct/kqapro.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model toolllama-2-7b --environment wiki --dataset kqapro >> data/logs/React/toolllama-2-7b/kqapro.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model gpt-3.5-turbo-1106 --environment wiki --dataset kqapro >> data/logs/React/gpt-3.5-turbo-1106/kqapro.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model gpt-4-1106-preview --environment wiki --dataset kqapro >> data/logs/React/gpt-4-1106-preview/kqapro.log &

