'''
Train
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from CoTAttack import CoTAttack
from MyModel import MyModel

if __name__ == '__main__':
    model_path = "./models/Llama-3.2V-11B-cot"
    victim_model = MyModel(model_path)
    '''
    # Test Call Function:
    image = Image.open("./images/math-geometry-2.png")
    messages = [
        {"role": "user", 
        "content": [
            {"type": "image"},
            {"type": "text", 
            "text": "If the line m is parallel to the side AB, what is angle a? You know the answer is A option. Options: A. 130o; B. 25o; C. 65o; D. 50o"}
        ]}
    ]
    output_tensor, output_Text = victim_model([image], messages)'
    '''
    prompt_temp_path = "./prompt_temp.txt"
    Attack = CoTAttack(victim_model, prompt_temp_path)
    