'''
Generated by Hunyuan T1. Code using PGD on CLIP
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np

# 示例模型加载（以CLIP为例）
class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = torch.hub.load('pytorch/vision:v1.10.0', 'clip', pretrained=' ViTBERT-G')
        self.clip.eval()

    def forward(self, images):
        return self.clip(images)

# 对抗攻击类
class VisualAdversarialAttack:
    def __init__(self, model, image_size=(224, 224), epsilon=0.05, num_steps=20, alpha=2):
        self.model = model
        self.image_size = image_size
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
        self.device = next(iter(model.parameters())).device if model else 'cpu'
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        
    def attack(self, image_path):
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).to(self.device)
        original_image = image_tensor.clone().detach()
        
        # 初始化扰动
        perturbation = torch.zeros_like(image_tensor, requires_grad=True)
        
        for _ in range(self.num_steps):
            image = image_tensor + perturbation
            image.requires_grad = True
            
            outputs = self.model(image)
            _, predicted = torch.max(outputs, dim=1)
            target_label = predicted.cpu().numpy()[0]
            
            loss = nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
            loss.backward()
            gradient = image.grad.data
            gradient_sign = gradient.sign()
            perturbation += self.alpha * gradient_sign
            
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            image.grad = None
        
        adversarial_image = original_image + perturbation
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
        
        return adversarial_image

# 使用示例
if __name__ == "__main__":
    # 加载模型
    model = CLIPModel()
    
    # 创建攻击实例
    attacker = VisualAdversarialAttack(model, epsilon=0.05, num_steps=20)
    
    # 攻击图像
    adversarial_image = attacker.attack("input_image.jpg")
    
    # 显示结果
    adversarial_image_pil = transforms.ToPILImage()(adversarial_image.cpu())
    adversarial_image_pil.show()