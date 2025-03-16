import argparse
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
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close() if filename else None

def main():
    # Parseo de argumentos
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a pre-train-model to keep training it')
    parser.add_argument('--variability', type=float, default=0.0,
                        help='Initial variability value (0-10)')
    parser.add_argument('--max_variability', type=float, default=50.0,
                        help='Max cariability')
    parser.add_argument('--variability_step', type=float, default=0.2,
                        help='Variability increase per variability interval')
    parser.add_argument('--variability_interval', type=int, default=50,
                        help='Episodes neede for an increase in variability interval')
    args = parser.parse_args()

    # Configuración
    EPISODES = 5000
    MAX_STEPS = 500
    UPDATE_INTERVAL = 1750
    SAVE_INTERVAL = 250
    PLOT_INTERVAL = 100

    os.makedirs("training_plots", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # Gradually increase variability
    current_variability = args.variability
    start_episode = 0

    env = environment.CollectorEnv(render=False)

    # In case of a pre-trained model
    if args.model_path:
        agent = PPOAgent(env)
        start_episode, metadata = agent.load_model(args.model_path)
        print(f"Loading model from path: {args.model_path}")
        # Look at the variability
        if metadata and 'variability' in metadata:
            current_variability = metadata['variability']
            print(f"Variability: {current_variability}")
        # Policy as train to keep training it
        agent.policy.train()
    else:
        agent = PPOAgent(
            env,
            hidden_size=512,
            lr_actor=0.001,
            lr_critic=0.001,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            gamma=0.97
        )

    global episode_rewards, moving_avg_rewards
    episode_rewards = []
    moving_avg_rewards = []
    total_steps = 0

    # Training loop for PPO
    for episode in range(start_episode, start_episode + EPISODES):
        # Variability increase
        if episode > start_episode and episode % args.variability_interval == 0 and current_variability < args.max_variability:
            current_variability = min(current_variability + args.variability_step, args.max_variability)
            print(f"Episodio {episode}, variabilidad incrementada a: {current_variability:.2f}")
        
        # Reset the enviroment
        state = env.reset(variability=current_variability)
        episode_reward = 0
        done = False
        print(f'Iniciando episodio {episode} con variabilidad {current_variability:.2f}')

        for _ in range(MAX_STEPS):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience (Buffer)
            agent.buffer.store(state, action, reward, value, log_prob.item(), done)
            
            episode_reward += reward
            total_steps += 1
            state = next_state
            
            # Policy Update
            if total_steps % UPDATE_INTERVAL == 0:
                agent.update()
            
            if done:
                break

        episode_rewards.append(episode_reward)
        moving_avg = np.mean(episode_rewards[-100:]) 
        moving_avg_rewards.append(moving_avg)
        
        # Visualization
        if (episode + 1) % PLOT_INTERVAL == 0:
            plot_training_progress(f"training_plots/progress_ep{episode+1}_var{current_variability:.1f}.png")
        
        # Save Model
        if (episode + 1) % SAVE_INTERVAL == 0:
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            metadata = {'variability': current_variability}
            agent.save_model(f"{model_dir}/ppo_ep{episode+1}_var{current_variability:.1f}.pth", episode+1, metadata)
            print(f"Model saved at {model_dir}/ppo_ep{episode+1}_var{current_variability:.1f}.pth")

    # Evaluation with different variability
    print("\nEvaluation:")
    for var in [0, 5, 10]:
        test_env = environment.CollectorEnv(render=True)
        state = test_env.reset(variability=var)
        total_reward = 0
        print(f"\nVariability {var}:")
        
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

if __name__ == '__main__':
    main()
