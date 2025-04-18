'''
This is an attack environment of interaction with victim models
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from typing import List

class AttackEnv:
    def __init__(self, 
                 victim_model, 
                 text_encoder,
                 prompt_temp_path):
        self.victim_model = victim_model
        self.text_encoder = text_encoder
        self.prompt_repo = self.init_prompt_dataset(prompt_temp_path)
    
    def init_prompt_dataset(self, prompt_temp_path) -> List[str]:  
        '''
        Load from txt directly, which the content in the txt is planned
        to be generated by LLMs(gpt-4o, etc.)
        '''
        with open(prompt_temp_path, 'r', encoding='utf-8') as file:
            raw_txt = file.read()
    
        prompt_temps = [p for p in raw_txt.split('\n\n') if p]
        return prompt_temps
    
    def _get_prompt_dataset(self) -> List[str]:
        return self.prompt_repo
    
    def reset(self):
        '''
        Note: 
        because the embedding cannot be converted into natural text again, 
        at least it is difficult to some extent,
        it is not very suitable to return embedding in RESET
        '''
        random_index = torch.randint(0, len(self.prompt_repo), (1,)).item()

        text_embedding, _ = self.text_encoder(prompt = self.prompt_repo[random_index])

        return text_embedding
    
    def step(self):
        pass