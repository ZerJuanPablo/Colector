# Collector with Policy Gradient

**"Team_Two" Reinforcement Learning Project using Proximal Policy Optimization (PPO)**

---

## Introduction

In this project, we demonstrate the use of Proximal Policy Optimization (PPO), an advanced policy gradient algorithm, applied within a custom-built PyGame environment. Our primary goal is to explore key concepts in reinforcement learning, implement the PPO algorithm from scratch, and evaluate interactions between our learning agent and its environment through practical experimentation.

---

## Repository Structure

- **`/src`**: Contains the source code necessary for PPO, the environment, training loops, and evaluations.
  - `agent.py`: PPO agent implementation including policy definition, action selection, and model checkpoint handling.
  - `environment.py`: Defines the custom environment logic, handling interactions with reward items (balls) and penalty items (traps).
  - `train.py`: Manages training loop and hyperparameter configurations.
  - `evaluate.py`: Runs trained agent evaluations.
  - `testing.py`: Facilitates manual testing of the environment without RL intervention.
  - `plot_results.py`: Generates visual representations of training outcomes.

- `/saved_models`: Stores trained models.
- `/training_plots`: Stores visualizations of training performance.
- `/deprecated`: Contains older versions of scripts no longer actively used.

---

## Running Instructions

### Training and Evaluation Instructions

#### Training
To train the PPO agent, execute the `train.py` script with optional parameters to control the variability of the environment:

```bash
python src/train.py [--variability 0] [--max_variability 50] [--variability_step 0.5] [--variability_interval 25]
```
#### Evaluation
To evaluate a trained model checkpoint, run:
```bash
python src/evaluate.py <path_to_saved_model>
```

---

## Environment Description

Our custom environment simulates a 2D collector scenario where our agent collects rewards (balls) while avoiding penalties (traps).

- **Dimensions and Rendering**:  
  - 800x600 pixel window, visualized using PyGame.

- **Agent**:
  - Square (40px), constant speed (5 px/step).
  - Spawned near the center with slight randomness.

- **Objects**:
  - **Balls (Rewards)**: 10 balls (10-pixel radius); rewards exponentially increase with each collected ball.
  - **Traps**: 3 traps (radius 15 pixels), collisions trigger a penalty and initiate a 30-step cooldown.

- **Reward Structure**:
  - Step Penalty: `-0.5` per step.
  - Ball Rewards: Progressive geometric scaling.
  - Trap Penalty: `-50`.
  - Wall Penalty: `-0.3`.
  - Shaping Reward (0.08): Encourages exploration and helps avoid local optima.
  - Finish Reward: `+200`.

- **Observation Space (Sensors)**:
  - 8 ball sensors positioned every 45 degrees around the agent, each providing normalized proximity readings (0-1 range).
  - 4 directional trap sensors (north, east, south, west), indicating proximity to traps.
  - A progress indicator representing the proportion of balls collected.

Code snippet demonstrating observation:

```python
def _get_observation(self):
    ball_sensors = self._get_ball_sensors()
    trap_sensors = self._get_trap_sensors()
    progress = 1.0 - (np.sum(self.active_balls) / self.NUM_BALLS)

    return np.concatenate([
        ball_sensors,
        trap_sensors,
        [progress]
    ])
```

- **Action Space**:
  - 5 discrete actions: stationary, left, right, up, down.

---
## PPO Algorithm Implementation

We chose PPO for its balance of stability and efficiency, using an Actor-Critic neural network architecture.

### Neural Network Structure

Our network contains:
- **Shared Layers**: Two fully connected layers (256 and 128 neurons with ReLU activation).
- **Actor**: Outputs action probabilities through a softmax layer.
- **Critic**: Estimates expected state values.

Example architecture:

```python
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
```
### Experience Collection and Policy Updates:
- Experiences were collected using Generalized Advantage Estimation (GAE) to improve training stability.
- PPO updates utilized:
  - Policy clipping with `ε = 0.2`
  - Entropy regularization (`coef = 0.02`) to enhance exploration.
  - Gradient clipping (`norm = 0.5`) to maintain training stability.

---
## Training and Experimentation

- We trained for 1000 episodes, each limited to 500 steps.
- Policy updates occurred every 2000 steps.

### Curriculum Strategy
- We started training in a deterministic environment, gradually introducing variability (0 to ±50 px randomness in object positions) to help generalize agent performance.

### Challenges Encountered:

- Initially, we used a pre-built PPO library (available here: [GitHub Link](https://github.com/Abraxius/ReinforcementLearning/tree/main)).The project works, but stayed incompleted as we needed to focus on creating the implementation of the algorithm ourselves.
- Key implementation errors included:
  - Initial fully randomized environment was too complex, making error detection difficult. Introducing controlled variability improved clarity to find other errors in the implementation.
  - Initially suboptimal observation space (direct coordinates) caused unstable learning, later resolved by introducing the simulation of sensors.
  - Incorrect computation of the critic loss led to catastrophic forgetting, requiring careful debugging and hyperparameter tuning.
  - The agent did not effectively learn to avoid traps. Hypothesized issues include improperly balanced penalties and rewards.
---

## Results

Progress visualizations at selected episodes:

| Episode | Visualization |
|----------|-------------|
| 1        | ![Episode 1](Documentation/Episode_1.gif) |
| 250      | ![Episode 250](Documentation/Episode_250.gif) |
| 500      | ![Episode 500](Documentation/Episode_500.gif) |
| 750      | ![Episode 750](Documentation/Episode_750.gif) |
| 1000     | ![Episode 1000](Documentation/Episode_1000.gif) |

The agent learned to collect all balls but consistently struggled with trap avoidance, suggesting a need for further reward tuning or extended training.

---

## Future Work

While the project concludes here, future improvements include:

- Revisiting the fully randomized environment, testing agent adaptability and robustness.
- Implement rewards to incentivize efficient movement patterns.
- Investigate and correct the issue of poor trap avoidance.
- Optimize training efficiency through code optimization and parallelization.

---

## References

- [Proximal Policy Optimization (PPO) by OpenAI](https://spinningup.openai.com/)
- Schulman et al. (2017), [PPO Original Paper](https://arxiv.org/abs/1707.06347)
