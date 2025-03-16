# Collector with Policy Gradient

"Team_Two" Policy Gradient Project for Reinforcement Learning

---
## Introduction

This project explores reinforcement learning (RL) by implementing a policy gradient method, specifically Policy Optimization (PO), within a custom collector environment built with PyGame. Our goal is to demonstrate a understanding of RL fundamentals, algorithmic implementation, and environment interaction.

---
## Repository Structure

The repository is organized as follows:

- **`/src`**: Contains all source code, including PPO implementation, environment definition, training loops, and evaluation scripts.
- **Main scripts in `/src`:**
  - `agent.py`: PPO implementation, agent creation, action selection, saving, and loading functionalities.
  - `environment.py`: Custom 2D environment setup with agent interactions, reward mechanisms, balls, and traps.
  - `train.py`: Main training loop, hyperparameter definitions, and environment-agent interactions.
  - `evaluate.py`: Evaluation routines for trained agents to assess performance.
  - `testing.py`: Manual testing of the environment without the RL agent.
  - `plot_results.py`: Script for plotting reward progression and monitoring training performance.

- `/saved_models`: Directory containing model checkpoints.
- `/training_plots`: Directory containing images reward progression visualization.
- `/deprecated`: Contains outdated scripts such as the previously used `main.py`.


---
## Environment Description

The environment is a custom 2D simulation built using PyGame, designed to emulate a collector scenario where an agent navigates to gather rewards while avoiding penalties. The main features and dynamics of the environment are as follows:

- **Dimensions and Rendering**:  
  The simulation area is set within an 800x600 pixel window. Rendering is optionally enabled using PyGame, allowing visualization of the agent, balls, and traps.

- **Agent Characteristics**:  
  The agent is represented as a square with a fixed size of 40 pixels and moves at a constant speed of 5 pixels per step. It is initialized near the center of the window with a random offset of up to ±50 pixels to introduce variability.

- **Reward Items (Balls)**:  
  - There are 10 balls in the environment, each with a radius of 10 pixels.
  - The ball positions are fixed (with an option to add variability during reset) to accelerate training.
  - When the agent collides with a ball, the ball is marked as collected and is deactivated for the remainder of the episode.

- **Penalty Items (Traps)**:  
  - The environment includes 3 traps, each with a radius of 15 pixels.
  - Colliding with a trap results in a penalty, and a cooldown period of 30 steps is initiated to prevent consecutive penalties.

- **Reward Structure**:
  - **Step Penalty (-0.1)**: Encourages time-efficient behavior while being small enough to avoid overwhelming other rewards
    
- **Shaping Reward Coefficient (0.08)**: Implemented as guidance to stop creation of local optima
    
- **Progressive Ball Rewards**: Geometric scaling (1.5 multiplier) creates an explicit curriculum - later collections become exponentially more valuable
    
- **Trap Penalty (-20)**: Sufficiently large to deter exploration near traps but smaller than episode failure penalties

- **Observation Space**:  
  Normalized coordinates (0-1 range) ensure stable neural network training. Relative position encoding reduces the need for the agent to learn absolute spatial reasoning. This includes:
  - The normalized position of the agent.
  - Relative normalized vectors and distances from the agent to each ball.
  - Relative normalized vectors and distances from the agent to each trap.
  - A progress indicator representing the proportion of balls collected.

- **Action Space**:  
  The agent can choose among 5 discrete actions, corresponding to remaining stationary or moving left, right, up, or down.


---
## Learning Algorithm: Proximal Policy Optimization (PPO)

The agent is trained using Proximal Policy Optimization (PPO), a policy gradient method recognized for its stability and sample efficiency.
### 1. Neural Network Architecture

Our PPO implementation utilizes an Actor-Critic architecture. The network is composed of shared layers that feed into two separate branches:

- **Shared Layers**:  
  The input observation (size 42) is processed through two fully-connected layers:
  - A linear layer transforming the input to a hidden representation, followed by a ReLU activation.
  - A second linear layer with 512 neurons and another ReLU activation.
  
- **Actor Branch**:  
  This branch outputs a probability distribution over the available actions:
  - A linear layer that reduces the 512-dimensional representation to 256 neurons, followed by ReLU.
  - A final linear layer produces a vector of size equal to the number of actions.
  - A softmax function is applied to obtain normalized action probabilities.

- **Critic Branch**:  
  This branch estimates the state value:
  - A linear layer from the shared representation to 256 neurons with ReLU activation.
  - A final linear layer outputs a single scalar representing the value of the state.

Our implementation:

```python
class PPONetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512),
            nn.ReLU()
        )
        # Actor branch
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        # Critic branch
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        x = self.shared(x)
        return torch.softmax(self.actor(x), dim=-1), self.critic(x).squeeze()
```

### 2. Experience Buffer and Advantage Estimation

The **PPOBuffer** class collects experiences (states, actions, rewards, values, log probabilities, and done flags) during training. Once a trajectory is complete, it computes advantages using **Generalized Advantage Estimation (GAE)** to balance bias and variance. The computed advantages are then normalized to stabilize learning.
Our implementation:
```python
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
```
### 3. PPO Agent and Policy Update

The **PPOAgent** class integrates the network and buffer to manage action selection, model saving/loading, and policy updates.

- **Action Selection**:  
    The agent converts the state into a tensor, passes it through the network, and samples an action from the resulting probability distribution. The corresponding log probability and state value are stored.
    
- **Policy Update**:  
    The update process is performed over mini-batches:
    
    - **Ratio Calculation**: The ratio of new to old policy probabilities for the taken actions is computed.
    - **Clipped Surrogate Objective**: Two surrogate loss terms (with and without clipping using a threshold `clip_epsilon`) are calculated; the minimum of these is used as the actor loss.
    - **Critic Loss**: Mean squared error loss is computed between the predicted values and the normalized advantages.
    - **Entropy Bonus**: An entropy term is included to encourage exploration by penalizing over-confident predictions.
    - **Gradient Clipping**: Gradients are clipped (with a maximum norm of 0.5) to prevent exploding gradients.
    - The network parameters are updated using the Adam optimizer with separate learning rates for the actor and critic.
Our implementation:
```python
class PPOAgent:
    def __init__(self, env, hidden_size=512, lr_actor=0.001, lr_critic=0.001, 
                 gamma=0.97, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(42, hidden_size, env.action_space).to(self.device)
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
        dist = Categorical(probs)
        action = dist.sample()
        self.action_history.append(action.item())
        return action.item(), dist.log_prob(action), value.item()
    
    def save_model(self, path, episode, metadata=None):
        if metadata is None:
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
            for idx in torch.randperm(len(states)).split(64):  # Mini-batch size of 64
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = normalized_advantages[idx]
                
                # Compute new probabilities and state values
                new_probs, new_values = self.policy(batch_states)
                dist = Categorical(new_probs)
                entropy = dist.entropy().mean()
                
                # Calculate the ratio for the PPO objective
                ratio = (dist.log_prob(batch_actions) - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(new_values, normalized_advantages[idx])
                
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        self.buffer.clear()
```
### 4. Training Loop and Evaluation

The training loop orchestrates the collection of experiences, policy updates, and periodic evaluation:

- **Experience Collection**:  
    During each episode, the agent interacts with the environment. Each transition (state, action, reward, etc.) is stored in the buffer.
- **Policy Updates**:  
    After a defined number of steps, the agent updates its policy using mini-batch updates over the collected data.
- **Checkpointing and Logging**:  
    Model checkpoints are saved at regular intervals, and training progress is visualized using reward plots.

```python
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
```
---
## Theoretical Background
### 1. Policy Gradient Methods

At the heart of many reinforcement learning algorithms is the idea of directly optimizing the policy—the function that maps states to actions—by gradient ascent on expected returns. The foundational formula for policy gradient methods, as used in the REINFORCE algorithm, is:

$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[\nabla_{\theta} \log \pi_{\theta}(a|s) \, Q^{\pi}(s, a)\right]
$$

Here, $J(\theta)$ represents the expected return when following the policy $\pi_{\theta}$  parameterized by $\theta$. The gradient $\nabla_{\theta} \log \pi_{\theta}(a|s)$  measures how a small change in the parameters affects the log-probability of taking action $a$ in state $s$, while $Q^{\pi}(s, a)$ is the action-value function estimating the future reward from taking action $a$ in state $s$.
### 2. Actor-Critic Methods

Our approach utilizes an **actor-critic architecture**. In this setup:
- The **actor** updates the policy directly by learning the probability distribution over actions.
- The **critic** evaluates the policy by estimating the state-value function $V(s)$.

This combination helps in reducing the variance of the policy gradient estimates, leading to more stable learning. However, standard actor-critic methods can still suffer from large, unstable updates.

### 3. Proximal Policy Optimization (PPO)

PPO is designed to address the instability inherent in policy gradient methods. Instead of performing large, unconstrained updates, PPO introduces a **clipped surrogate objective** to ensure that policy updates do not deviate too much from the current policy. The objective function in PPO is formulated as follows:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}\left(r_t(\theta), 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t \right) \right]
$$

Here:
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio between the new and the old policies.
- $\hat{A}_t$ is the estimated advantage at time step $t$.
- $\epsilon$ is a small hyperparameter that defines how far the new policy is allowed to deviate from the old policy.

The clipping function ensures that the ratio $r_t(\theta)$ remains within the interval $[1-\epsilon, 1+\epsilon]$. This prevents the update from being too large, thereby stabilizing the training process.

### 4. Automatic Differentiation Tools

To efficiently compute gradients and perform backpropagation through the neural network, our implementation uses **PyTorch**. 
### 5. Project Constraints

An important aspect of our project is that we deliberately avoid using pre-built implementations of PPO. This choice forces us to implement and understand every component of the algorithm from scratch, deepening our comprehension of both the theoretical and practical aspects of reinforcement learning. By building the PPO algorithm manually, we gain valuable insights into:
- The intricacies of the policy gradient update.
- The challenges in maintaining stability during training.
- The importance of techniques such as clipping and advantage normalization.

Overall, our approach combines a solid theoretical foundation with practical enhancements to deliver a robust and stable learning algorithm. This balance of theory and practice is essential for achieving effective reinforcement learning in complex environments.

---

## Training Methodology

### Training Procedure and Parameters

#### Training configuration
```python
EPISODES = 5000
MAX_STEPS = 500
UPDATE_INTERVAL = 1750  # Steps between policy updates
SAVE_INTERVAL = 250     # Checkpoint frequency
PLOT_INTERVAL = 100     # Progress visualization
```

- **5000 Episodes**: Balances training duration (≈3.5M steps) with computational constraints
    
- **500 Step Episodes**: Allows sufficient time for strategic behavior while preventing endless random exploration
    
- **1750 Update Interval**: Processes ~3.5 episodes worth of experience per update (batch size 64 → 27 updates/batch)

#### Curriculum Learning Strategy
The first approach to the environment was a fully random one, but the agent never learn a policy in a fully random environment. So we went for a deterministic environment, however the agent quickly felt into a local maxima and couldn't get all balls. To fix this we tried implementing a progressive difficulty adding a variability slowly. This made the policy more robust to variability. However the agent didn't stop falling for the same local maxima.
- **Progressive Difficulty**: Starts with deterministic ball positions (variability=0), gradually introducing positional noise
    
- **Smooth Transitions**: 0.2 step size prevents abrupt environment changes that could destabilize learning
    
- **Final Variability (50)**: Represents ±50 pixel random offset from default positions, testing generalization

#### PPO Hyperparameters
This was the hardest part to define as even the slight variation in them made the agent have an unstable learnrate. At the end we couldn't find better parameters, but they probably exist.

**Parameter Analysis**:

| Parameter   | Value | Role                           | Tradeoff Managed               | Reasoning behind                                                                                                                         |
| ----------- | ----- | ------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| γ (gamma)   | 0.97  | Discount future rewards        | Immediate vs long-term rewards | We tried to find a value that made the agent grab the closest ball but also trying to grab balls even when they aren't as close anymore. |
| λ (GAE)     | 0.95  | Advantage estimation smoothing | Bias-variance tradeoff         |                                                                                                                                          |
| ε (clip)    | 0.2   | Policy update constraint       | Exploration vs exploitation    | We tried to force the agent to explore more, as it never tried to go for all balls.                                                      |
| β (entropy) | 0.01  | Exploration incentive\|        | Policy diversity vs focus\|    | We tried to force the agent to explore more, as it never tried to go for all balls.                                                      |
**Update Mechanics**:
- **10 Epochs**: Tried to maximize data utilization without overfitting
- **Gradient Clipping (0.5)**: Prevents exploding gradients in deep network
- 
---

## Results

Key observations include:

- Description and interpretation of reward curves and performance metrics.
- Analysis of learning behavior, including successes and failures.

---

## Discussion and Analysis

### Project Insights

- Reflection on the training outcomes, mentioning limitations such as the agent reaching a local optimum and stopping improvement.
- Challenges encountered throughout implementation.

### Future Work

Suggestions on potential improvements or next steps, including algorithmic enhancements, environment complexity adjustments, or alternative RL strategies.

---

## References
1. Schulman et al. (2017) - [PPO Original Paper](https://arxiv.org/abs/1707.06347)
2. OpenAI Spinning Up - [PPO Implementation Guide](https://spinningup.openai.com/)

