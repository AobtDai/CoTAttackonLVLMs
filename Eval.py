import os
from openai import AzureOpenAI, OpenAIError

import yaml
from typing import Dict, Any

def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        return config
    
config = load_config("./TxtSet/apikey.yml")

os.environ["AZURE_OPENAI_API_KEY"] = config["api"]["key"]
os.environ["AZURE_OPENAI_ENDPOINT"] = config["api"]["url"]

def get_llm_response(msg, system_prompt:str=None):
     client = AzureOpenAI(
         azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
         api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
         api_version=config["api"]["api_version"]
     )
     if not system_prompt:
        with open("./TxtSet/Text_Aware_Eval.txt", "r", encoding="utf-8") as file:
            system_prompt = file.read()
 
     response = client.chat.completions.create(
         model="gpt-4o",
         messages=[
             { 
                 "role":"system",
                 "content":system_prompt
             },
            {
            "role": "user", 
            "content": [
	            {
	                "type": "text",
	                # "text": "Describe this picture:"
	                "text": msg
	            },
	            # {
	            #     "type": "image_url",
	            #     "image_url": {
                #         "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                #     }
                # } 
                ] 
            }
         ],
         max_tokens=1024,
        #  reasoning_effort = "medium"
    )
     return response.choices[0].message.content

with open("./TxtSet/user_input.txt", "r", encoding="utf-8") as file:
    user_input = file.read()
msg = user_input
# "Summarize several advatages to use RLHF (like GRPO) to train/finetune a LLM."
output = get_llm_response(msg=msg)
print(output)