nohup python3 pred.py --agent_name React_wiki_run_Agent --model vicuna-13b --environment wiki --dataset cpj >> data/logs/React/vicuna-13b/cpj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-13b --environment wiki --dataset cpj >> data/logs/React/llama2-13b/cpj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model tulu-7b --environment wiki --dataset cpj >> data/logs/React/tulu-7b/cpj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-7b-chat-4k --environment wiki --dataset cpj >> data/logs/React/llama2-7b-chat-4k/cpj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model codellama-13b-instruct --environment wiki --dataset cpj --num_workers 1 >> data/logs/React/codellama-13b-instruct/cpj.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model toolllama-2-7b --environment wiki --dataset cpj >> data/logs/React/toolllama-2-7b/cpj.log &


