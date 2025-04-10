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
    victim_model_path = "./models/Llama-3.2V-11B-cot"
    text_encoder_path = "./models/clip-vit-base-patch32"
    victim_model = MyModel(victim_model_path)
    text_encoder = MyModel(text_encoder_path)
    '''
    # Test Call Function:
    image = Image.open("./images/math-geometry-2.png")
    prompt = "If the line m is parallel to the side AB, what is angle a? You know the answer is A option. Options: A. 130o; B. 25o; C. 65o; D. 50o"
    output_tensor, output_Text = victim_model([image], prompt)
    '''
    prompt_temp_path = "./prompt_temp.txt"
    Attack = CoTAttack(victim_model, 
                       text_encoder, 
                       prompt_temp_path)
    