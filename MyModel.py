import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import List

class MyModel:
    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def __call__(self, images: list, messages):
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=10000)
        return output, self.processor.decode(output[0])