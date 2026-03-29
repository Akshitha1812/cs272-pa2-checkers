import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=36, action_dim=1296):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Policy head (Actor)
        self.actor_head = nn.Linear(128, action_dim)
        
        # Value head (Critic) 
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value

class ACAgent:
    def __init__(self, obs_dim=36, action_dim=1296, lr=1e-3, gamma=0.99):
        self.model = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def get_action(self, obs, action_mask):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        logits, _ = self.model(obs_tensor)
        
        # Apply mask: set invalid actions to very negative number
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
        masked_logits = logits.masked_fill(mask_tensor == 0, -1e9)
        
        probs = F.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action)

    def update(self, rewards, log_probs, values, next_values, dones):
        self.optimizer.zero_grad()
        
        actor_loss = 0
        critic_loss = 0
        
        # Calculate returns and advantages
        for i in range(len(rewards)):
            R = rewards[i] + self.gamma * next_values[i].item() * (1 - dones[i])
            target = torch.tensor([[R]], dtype=torch.float32)
            advantage = (target.squeeze() - values[i].squeeze()).detach()
            
            actor_loss -= log_probs[i] * advantage
            critic_loss += F.smooth_l1_loss(values[i], target)
            
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
