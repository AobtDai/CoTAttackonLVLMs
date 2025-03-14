import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from transformers import MllamaForConditionalGeneration, AutoProcessor
from CoTAttack import CoTAttack

if __name__ == '__main__':
    model_id = "./models/Llama-3.2V-11B-cot"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    Attack = CoTAttack(model, processor)
    
    