import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16
from typing import Dict, Any, List

# 定义环境类（模拟与受害者模型的交互）
class AttackEnvironment:
    def __init__(self, victim_model):
        self.victim_model = victim_model
        self.state_dim = 768  # 假设使用ViT-B/16提取的特征维度
        self.action_dim = 768  # 动作空间维度（与特征维度一致）
        self.max_episode_length = 10
    
    def reset(self) -> torch.Tensor:
        """重置环境，返回初始状态"""
        # 这里假设初始状态为随机向量，实际应使用真实提示词编码
        return torch.randn(1, self.state_dim)
    
    def step(self, action: torch.Tensor) -> tuple:
        """执行动作，返回新状态、奖励、完成标志"""
        # 将动作解码为文本提示词（此处为示例，需实际解码逻辑）
        perturbed_prompt = self.decode_action(action)
        
        # 执行红队攻击获取模型响应
        response = self.victim_model(perturbed_prompt)
        
        # 计算毒性奖励（示例使用随机值，需替换为真实检测逻辑）
        toxicity_score = random.uniform(0, 10)
        reward = self.calculate_reward(toxicity_score)
        
        # 检查终止条件（示例使用固定步数，需根据实际情况调整）
        done = (random.random() < 0.1)  # 10%概率终止
        
        # 返回新状态（示例使用随机特征，需替换为真实编码）
        new_state = torch.randn(1, self.state_dim)
        return new_state, reward, done, {}
    
    def decode_action(self, action: torch.Tensor) -> str:
        """将动作向量解码为文本提示词（示例占位符）"""
        return "optimized_prompt_" + str(action.argmax())
    
    def calculate_reward(self, toxicity_score: float) -> float:
        """计算奖励函数（示例：毒性越高奖励越高）"""
        return toxicity_score / 10.0

# 定义ViT特征提取器
class ViTFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        self.fc = nn.Linear(768, output_dim)  # 使用ViT原生的768维度
        
    def forward(self, x):
        features = self.vit(x)[1]  # 使用CLS token特征
        return self.fc(features)

# Define DDPG Actor using Transformer
class TransformerActor(nn.Module): 
    '''
    Output can be a piece of optimized text embedding or tokens.
    Cause the length of both input and output are unknown, models are limited.
    Let me find out some pre-trained models, which I just use its ability in semantics.
    However, malicious inputs will not be limited to semantic-related in further experiments.
    '''
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.embedding = nn.Linear(state_dim, 128)
        self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.fc = nn.Linear(128, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.embedding(state).unsqueeze(0)  # Add batch dimension
        x = self.transformer(x).squeeze(0)  # Apply transformer
        return self.tanh(self.fc(x))  # Output action in range [-1,1]

# Define DDPG Critic
class DDPGCritic(nn.Module):
    '''
    Inputs include responses of victim models.
    Output are generated from responses.
    '''
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Q-value output

# 定义DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.actor = TransformerActor(state_dim, action_dim)
        self.critic = DDPGCritic(state_dim, action_dim)
        self.target_actor = TransformerActor(state_dim, action_dim)
        self.target_critic = DDPGCritic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = []  # 经验回放缓冲区（需扩展实现）
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.005  # 软更新参数
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """选择动作（带噪声探索）"""
        with torch.no_grad():
            action = self.actor(state).numpy()
        # 添加高斯噪声（示例，需根据需求调整）
        noise = torch.randn_like(action) * 0.1
        return torch.clamp(action + noise, -1, 1)
    
    def update(self):
        """更新网络参数"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从经验池采样（示例，需实现具体采样逻辑）
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        dones = torch.cat(dones)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + self.gamma * self.target_critic(next_states, next_actions) * (1 - dones)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# 定义对抗攻击管道
class AttackPipeline:
    def __init__(self, victim_model):
        self.victim_model = victim_model
        self.env = AttackEnvironment(victim_model)
        self.agent = DDPGAgent(state_dim=self.env.state_dim, action_dim=self.env.action_dim)
        self.prompt_repo = self.init_prompt_dataset()
        self.attack_round = 0
    
    def init_prompt_dataset(self) -> List[str]:
        """初始化提示词库（示例使用静态列表）"""
        return [
            "This is a benign prompt.",
            "Let's try something harmful.",
            "Generate offensive content."
        ]
    
    def generate_adversarial_prompt(self, base_prompt: str) -> str:
        """使用强化学习生成对抗性提示词"""
        state = self.env.reset()
        for _ in range(self.env.max_episode_length):
            action = self.agent.select_action(state.unsqueeze(0))
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            if done:
                break
        return self.env.decode_action(action)
    
    def perturbation_optimization(self, prompt: str) -> str:
        """扰动优化（集成到DDPG流程中）"""
        return self.generate_adversarial_prompt(prompt)
    
    def toxicity_detection(self, output: str) -> Dict[str, Any]:
        """毒性检测（示例使用随机评分）"""
        score = random.randint(1, 10)
        return {
            "is_toxic": score > 5,
            "toxic_score": score / 10.0,
            "diversity_metric": random.random(),
            "correlation_metric": random.random()
        }
    
    def red_team_attack(self, prompt: str) -> str:
        """红队攻击执行"""
        return self.victim_model(prompt)
    
    def run_attack(self, base_prompt: str) -> Dict[str, Any]:
        """执行完整攻击流程"""
        adversarial_prompt = self.generate_adversarial_prompt(base_prompt)
        optimized_prompt = self.perturbation_optimization(adversarial_prompt)
        model_output = self.red_team_attack(optimized_prompt)
        detection_result = self.toxicity_detection(model_output)
        success = detection_result["is_toxic"]
        
        # 存储经验到记忆库（示例）
        self.agent.memory.append((
            torch.tensor([0.0]),  # 状态
            torch.tensor([0.0]),  # 动作
            torch.tensor([0.0]),  # 奖励
            torch.tensor([0.0]),  # 下一状态
            torch.tensor([0.0])   # 完成标志
        ))
        
        self.attack_round += 1
        return {
            "success": success,
            "response": model_output,
            "detection": detection_result,
            "round": self.attack_round
        }

# 示例受害者模型（需替换为真实模型）
class VictimModel:
    def __call__(self, prompt: str) -> str:
        """模拟受害者模型响应"""
        return f"Response to: {prompt}"

if __name__ == "__main__":
    victim_model = VictimModel()
    
    pipeline = AttackPipeline(victim_model)
    
    # 执行攻击
    results = []
    for _ in range(5):
        result = pipeline.run_attack(pipeline.prompt_repo[0])
        results.append(result)
        print(f"\nAttack Round {_+1}:")
        print(f"Success: {result['success']}")
        print(f"Response: {result['response']}")
        print(f"Toxicity: {result['detection']}")
    
    # 训练强化学习代理（示例）
    for episode in range(3):
        state = pipeline.env.reset()
        done = False
        while not done:
            action = pipeline.agent.select_action(state.unsqueeze(0))
            next_state, reward, done, _ = pipeline.env.step(action)
            pipeline.agent.memory.append((state, action, reward, next_state, done))
            state = next_state
        pipeline.agent.update()