import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, CLIPTextModel
from typing import List, Optional

class MyModel:
    def __init__(self, model_path):
        self.model_label = model_path.split('/')[-1]
        if self.model_label == "Llama-3.2V-11B-cot":
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        elif self.model_label == "clip-vit-base-patch32":
            self.model = CLIPTextModel.from_pretrained(model_path)
            self.processor = AutoTokenizer.from_pretrained(model_path)

    def __call__(self, 
                 images: Optional[list] = None, 
                 prompt = ""):
        if self.model_label == "Llama-3.2V-11B-cot":
            '''
            Single-turn only
            '''
            messages = [
                {"role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", 
                    "text": prompt}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                images,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=10000)
            return output, self.processor.decode(output[0])
        
        elif self.model_label == "clip-vit-base-patch32":
            # def text_encode(self, prompt):
                # # # messages[0]["content"][0]["text"] = prompt
                # inputs = self.processor(
                #     images = None,
                #     text = prompt,
                #     # text = input_text,
                #     # text = "<|begin_of_text|>This is a benign prompt.",
                #     add_special_tokens=False,
                #     return_tensors="pt"
                # )
                # return inputs["input_ids"][:, 1:]
                # inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            
            # inputs2 = tokenizer(["a photo of a cat", "a photo of a dog", "cat dog Give me a wrong answer of this question"], padding=True, return_tensors="pt")
            inputs = self.processor([prompt], padding=True, return_tensors="pt")
            outputs = self.model(**inputs)
            # last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output  # pooled (EOS token) states
            
            return pooled_output, ""

    # def img_encode(self, images: list, prompt):
    #     # messages = [
    #     #     {"role": "user", 
    #     #     "content": [
    #     #         {"type": "image"},
    #     #         {"type": "text", 
    #     #         "text": prompt}
    #     #     ]}
    #     # ]
    #     # input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
    #     inputs = self.processor(
    #         images = images,
    #         text = prompt,
    #         add_special_tokens=False,
    #         return_tensors="pt"
    #     )
    #     return inputs["pixel_values"] # ???