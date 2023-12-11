nohup python3 pred.py --agent_name React_wiki_run_Agent --model vicuna-13b --environment wiki --dataset musique >> data/logs/React/vicuna-13b/musique.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-13b --environment wiki --dataset musique >> data/logs/React/llama2-13b/musique.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model tulu-7b --environment wiki --dataset musique >> data/logs/React/tulu-7b/musique.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model llama2-7b-chat-4k --environment wiki --dataset musique >> data/logs/React/llama2-7b-chat-4k/musique.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model codellama-13b-instruct --environment wiki --dataset musique --num_workers 1 >> data/logs/React/codellama-13b-instruct/musique.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model toolllama-2-7b --environment wiki --dataset musique >> data/logs/React/toolllama-2-7b/musique.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model gpt-3.5-turbo-1106 --environment wiki --dataset musique >> data/logs/React/gpt-3.5-turbo-1106/musique.log &

nohup python3 pred.py --agent_name React_wiki_run_Agent --model gpt-4-1106-preview --environment wiki --dataset musique >> data/logs/React/gpt-4-1106-preview/musique.log &

