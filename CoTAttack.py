import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms

class CoTAttack:
    def __init__(self, victim_model, victim_processor):
        self.victim_model = victim_model
        self.victim_processor = victim_processor