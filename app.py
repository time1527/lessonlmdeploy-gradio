# https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/serving/gradio.md
import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-1_8b-4bit'
os.system(f'git clone https://code.openxlab.org.cn/q4171119/lesson-lmdeploy.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,
                                             load_in_4bit=True, 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.float16,
                                             device_map="auto")

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-1.8B-Self-Cognition",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
