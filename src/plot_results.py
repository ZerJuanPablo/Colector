import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_from_file(filename):
    data = torch.load(filename)
    
    plt.figure(figsize=(10, 5))
    
    # Rewards
    plt.plot(data['episode_rewards'], alpha=0.3, label='Reward per Episode')
    
    # Moving Average
    moving_avg = [np.mean(data['episode_rewards'][max(0, i-100):i+1]) 
                 for i in range(len(data['episode_rewards']))]
    plt.plot(moving_avg, linewidth=2, label='Moving Average (100 episodes)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f"Training Progress\nHyperparameters: {data['hyperparameters']}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_from_file("training_data.pth")
