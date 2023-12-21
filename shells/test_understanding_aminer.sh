nohup python3 pred.py --agent_name React_aminer_run_Agent --model vicuna-13b --environment aminer --dataset profiling >> data/logs/React/vicuna-13b/profiling.log &

nohup python3 pred.py --agent_name React_aminer_run_Agent --model llama2-13b --environment aminer --dataset profiling >> data/logs/React/llama2-13b/profiling.log &

nohup python3 pred.py --agent_name React_aminer_run_Agent --model tulu-7b --environment aminer --dataset profiling >> data/logs/React/tulu-7b/profiling.log &

nohup python3 pred.py --agent_name React_aminer_run_Agent --model llama2-7b-chat-4k --environment aminer --dataset profiling >> data/logs/React/llama2-7b-chat-4k/profiling.log &

nohup python3 pred.py --agent_name React_aminer_run_Agent --model codellama-13b-instruct --environment aminer --dataset profiling --num_workers 1 >> data/logs/React/codellama-13b-instruct/profiling.log &

nohup python3 pred.py --agent_name React_aminer_run_Agent --model toolllama-2-7b --environment aminer --dataset profiling >> data/logs/React/toolllama-2-7b/profiling.log &

nohup python3 pred.py --agent_name React_aminer_run_Agent --model gpt-3.5-turbo-1106 --environment aminer --dataset profiling >> data/logs/React/gpt-3.5-turbo-1106/profiling.log &

nohup python3 pred.py --agent_name React_aminer_run_Agent --model gpt-4-1106-preview --environment aminer --dataset profiling >> data/logs/React/gpt-4-1106-preview/profiling.log &

