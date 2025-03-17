import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Categorical
import torch.nn.functional as F

class PPONetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU()
        )
        
        # Actor branch
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        # Critic branch
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.shared(x)
        return torch.softmax(self.actor(x), dim=-1), self.critic(x).squeeze()

class PPOBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self):
        advantages = []
        last_advantage = 0
        next_value = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
            next_value = self.values[t]
            
        advantages = torch.tensor(advantages, dtype=torch.float32)
        return advantages, (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class PPOAgent:
    def __init__(self, env, hidden_size=256, lr_actor=0.0003, lr_critic=0.001, 
                 gamma=0.97, gae_lambda=0.95, clip_epsilon=0.25, entropy_coef=0.03):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(13, hidden_size, env.action_space).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.buffer = PPOBuffer(gamma, gae_lambda)
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.action_history = []

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probs, value = self.policy(state_tensor)

        # Add occasional random exploration
        if random.random() < 0.05:  # 5% random actions
            action = torch.randint(0, self.env.action_space, (1,)).to(self.device)
            dist = Categorical(probs)
            return action.item(), dist.log_prob(action), value.item()

        dist = Categorical(probs)
        action = dist.sample()
        self.action_history.append(action.item())
        return action.item(), dist.log_prob(action), value.item()
    
    def save_model(self, path, episode, metadata=None):
        if metadata == None:
            metadata = {}
        torch.save({
            'episode': episode,
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metadata': metadata
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint['episode'], checkpoint.get('metadata', {})

    def update(self):
        states = torch.FloatTensor(np.array(self.buffer.states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages, normalized_advantages = self.buffer.compute_advantages()
        normalized_advantages = normalized_advantages.float().to(self.device)
        
        for _ in range(10):
            for idx in torch.randperm(len(states)).split(256):  # Batch size 128
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = normalized_advantages[idx]
                
                # New probabilities
                new_probs, new_values = self.policy(batch_states)
                dist = Categorical(new_probs)
                entropy = dist.entropy().mean()
                
                # Ratio
                ratio = (dist.log_prob(batch_actions) - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(new_values, normalized_advantages[idx])
                
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        self.buffer.clear()