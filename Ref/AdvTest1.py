'''
Generated by Hunyuan T1. LVLMs
'''


import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoModelForVisionLanguageTasks, AutoTokenizer

class VisualAdversarialAttack:
    def __init__(self, model, epsilon=8/255, alpha=2/255, steps=30):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.loss_fn = nn.CrossEntropyLoss()
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 对抗空间归一化[6](@ref)
        ])
        
    def perturb(self, image, text_query):
        image.requires_grad_(True)
        perturbation = torch.zeros_like(image, requires_grad=True)
        
        for _ in range(self.steps):
            outputs = self.model(image + perturbation, text_query)
            loss = -outputs.logits.argmax(dim=-1).float().mean()  # 最大化损失[6](@ref)
            self.model.zero_grad()
            loss.backward()
            
            # 自适应梯度缩放[6](@ref)
            grad_norm = perturbation.grad.view(perturbation.shape[0], -1).norm(dim=1)
            scale = self.alpha / (grad_norm + 1e-8)
            perturbation.grad *= scale.view(-1, 1, 1, 1)
            
            perturbation.data += perturbation.grad * self.alpha
            perturbation.data = torch.clamp(perturbation.data, -self.epsilon, self.epsilon)
            perturbation.grad.zero_()
        
        return (image + perturbation).detach()

# 使用示例
model_name = "llama-7b-vqav2"  # 假设支持视觉输入的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForVisionLanguageTasks.from_pretrained(model_name).to("cuda")

image = Image.open("test.jpg").convert("RGB")
text_query = "a cat sitting on a mat"

attacker = VisualAdversarialAttack(model)
adversarial_image = attacker.perturb(image, text_query)

# 验证攻击效果
original_output = model(image, text_query).logits.argmax(dim=-1).item()
adversarial_output = model(adversarial_image, text_query).logits.argmax(dim=-1).item()

print(f"Original prediction: {tokenizer.decode(original_output)}")
print(f"Adversarial prediction: {tokenizer.decode(adversarial_output)}")