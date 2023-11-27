import torch, gc
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import json
import random
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
# ["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "internlm-7b-8k"]
# model_name = "llama2-7b-chat-4k"
model_names = ["llama2-7b-chat-4k", "tulu-7b"]

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# def parse_args(args=None):
#     parser = argparse.ArgumentParser()
#     # agent args
#     parser.add_argument('--model', type=str, default="tulu-7b", choices=["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "internlm-7b-8k"])
    
def load_model_and_tokenizer(path, model_name, device,  load_token_only=False):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if not load_token_only:
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                  output_scores=True, return_dict_in_generate=True, 
                                                  torch_dtype=torch.bfloat16).to(device)
            model.eval()
    elif "llama2" in model_name or "tulu" in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        if not load_token_only:
            model = LlamaForCausalLM.from_pretrained(path, output_scores=True, return_dict_in_generate=True, 
                                                 torch_dtype=torch.bfloat16).to(device) 
    elif "gpt-" in model_name:
        return None, None
    
    if load_token_only:
        return tokenizer
    else:
        model = model.eval()
        return model, tokenizer

seed_everything(42)
# args = parse_args()
model2path = json.load(open("../../config/model2path.json", "r"))
model2maxlen = json.load(open("../../config/model2maxlen.json", "r"))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define your model
models = {}
tokenizers = {}
for i, model_name in  enumerate(model_names):
    max_length = model2maxlen[model_name]
    device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    models[model_name] = model
    tokenizers[model_name] = tokenizer
app = FastAPI()



# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)      
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "tulu" in model_name:
        prompt = f"<|user|>:{prompt}\n<|assistant|>:"
    return prompt
  
def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def infer(prompt, max_gen, temperature, model_name):
    # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
    torch.cuda.empty_cache()
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    prompt = build_chat(tokenizer, prompt, model_name)
    _input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    outputs = model.generate(
                **_input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=temperature,
            )
    scores = outputs.scores
    output_ids = outputs.sequences[0, -len(scores):]
    pred = tokenizer.decode(output_ids, skip_special_tokens=True)
    pred = post_process(pred, model_name)
    return pred



class Item(BaseModel):
    input_text: str = None
    max_new_tokens: int = 100
    temperature: float = 1.0


@app.post('/completion/{model_name}')
def get_completion(model_name:str, request_data: Item):
    input_text = request_data.input_text
    max_new_tokens = request_data.max_new_tokens
    temperature = request_data.temperature
    completions_text = infer(input_text, max_new_tokens, temperature, model_name)
    res = {
        'completions_text': completions_text,
    }
    return res

