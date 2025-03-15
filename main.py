import environment
from agent import PPOAgent
import numpy as np
import torch
import os
import datetime
import matplotlib.pyplot as plt

def plot_training_progress(filename=None):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Reward per episode')
    plt.plot(moving_avg_rewards, linewidth=2, label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()

    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close() if filename else None

# Configuration
EPISODES = 3000
MAX_STEPS = 500
UPDATE_INTERVAL = 1750
SAVE_INTERVAL = 250
PLOT_INTERVAL = 100

os.makedirs("training_plots", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

env = environment.CollectorEnv(render=False)
agent = PPOAgent(
    env,
    hidden_size=512,
    lr_actor=0.001,
    lr_critic=0.001,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    gamma=0.97
)

episode_rewards = []
moving_avg_rewards = []
total_steps = 0

# PPO training loop
for episode in range(EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False
    print(f'Finished episode {episode}')

    for _ in range(MAX_STEPS):
        action, log_prob, value = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Store experience
        agent.buffer.store(state, action, reward, value, log_prob.item(), done)
        
        episode_reward += reward
        total_steps += 1
        state = next_state
        
        # Update policy
        if total_steps % UPDATE_INTERVAL == 0:
            agent.update()
        
        if done:
            break

    episode_rewards.append(episode_reward)
    moving_avg = np.mean(episode_rewards[-100:]) 
    moving_avg_rewards.append(moving_avg)
    
    # Visualization
    if (episode + 1) % PLOT_INTERVAL == 0:
        plot_training_progress(f"training_plots/progress_ep{episode+1}.png")
    
    # Save model
    if (episode + 1) % SAVE_INTERVAL == 0:
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        agent.save_model(f"{model_dir}/ppo_ep{episode+1}.pth", episode+1)

# Evaluation
test_env = environment.CollectorEnv(render=True)
state = test_env.reset()
total_reward = 0

for _ in range(MAX_STEPS):
    with torch.no_grad():
        probs, _ = agent.policy(torch.FloatTensor(state).to('cuda'))
    action = torch.argmax(probs).item()
    
    next_state, reward, done, _ = test_env.step(action)
    total_reward += reward
    state = next_state
    
    if done:
        break

print(f"Final reward: {total_reward}")
test_env.close()