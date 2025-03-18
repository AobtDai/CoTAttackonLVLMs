'''
This is an attack pipeline
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from AttackEnv import AttackEnv

class CoTAttack:
    def __init__(self, victim_model, prompt_temp_path):
        self.victim_model = victim_model
        self.env = AttackEnv(victim_model, prompt_temp_path)

    