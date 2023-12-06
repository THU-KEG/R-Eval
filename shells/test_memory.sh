nohup python3 pred.py --agent_name React_wiki_run_Agent --model vicuna-13b --environment wiki --dataset csj >> data/logs/React/vicuna-13b/csj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-13b --environment wiki --dataset csj >> data/logs/React/llama2-13b/csj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model tulu-7b --environment wiki --dataset csj >> data/logs/React/tulu-7b/csj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-7b-chat-4k --environment wiki --dataset csj >> data/logs/React/llama2-7b-chat-4k/csj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model codellama-13b-instruct --environment wiki --dataset csj --num_workers 1 >> data/logs/React/codellama-13b-instruct/csj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model toolllama-2-7b --environment wiki --dataset csj >> data/logs/React/toolllama-2-7b/csj.log &


