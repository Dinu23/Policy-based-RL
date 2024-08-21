# Policy-based Reinforcement Learning

## Overview

This repository focuses on the implementation and exploration of policy-based reinforcement learning algorithms, which directly parameterize and optimize the policy function instead of estimating value functions. These methods are applied to a custom reinforcement learning environment called `Catch`, where the goal is to move a paddle to catch falling balls.

In addition to standard policy gradient methods, this project includes the Covariance Matrix Adaptation Evolution Strategy (CMA-ES), a powerful evolutionary algorithm for optimizing non-convex functions, which can be applied as a training method in reinforcement learning tasks.

## Features

- **REINFORCE Algorithm**: A Monte Carlo-based policy gradient method that updates the policy parameters directly based on the return from each episode.
- **Actor-Critic Methods**:
  - **A2C (Advantage Actor-Critic)**: Combines value-based and policy-based methods by using a critic to estimate the value function and an actor to update the policy.
  - **PPO (Proximal Policy Optimization)**: A more stable and efficient policy gradient method that prevents large updates to the policy by constraining the optimization step.
- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: An evolutionary algorithm that optimizes the policy by adapting the covariance matrix of a multivariate normal distribution, effectively exploring the policy space.
- **Discrete Action Spaces**: The implementations in this repository are designed to handle environments with discrete action spaces. (However we the methods can be addapted to continous action spaces)

## The `Catch` Environment

### Description

The `Catch` environment is a reinforcement learning task where the agent controls a paddle that must catch balls dropping from the top of the screen. The environment is designed as follows:

- **State Space**:
  - The environment can be configured with adjustable width and height using the `rows` and `columns` parameters.
  - Observations can be either:
    - A vector representing the xy-locations of the paddle and the lowest ball.
    - A binary two-channel pixel array, where the first channel represents the paddle location and the second channel represents the balls' positions.
  - The type of observation space is determined by the `observation_type` parameter during initialization.

- **Action Space**:
  - The paddle can move left, right, or remain idle at each timestep.

- **Reward Function**:
  - +1 reward for catching a ball at the bottom row.
  - -1 penalty for missing a ball that reaches the bottom row.
  - 0 reward for all other situations.

- **Dynamics**:
  - Balls drop randomly from one of the possible positions at the top of the screen.
  - The dropping speed can be adjusted using the `speed` parameter.




## Key Components

### 1. REINFORCE
- **Overview**: A basic policy gradient method where the policy is updated using the total return from an episode.
- **Strengths**: Simple to implement and works well in environments where episodic rewards are sparse.

### 2. Advantage Actor-Critic (A2C)
- **Overview**: Combines the benefits of value-based and policy-based methods by estimating both a value function (critic) and a policy (actor).
- **Strengths**: Provides lower variance in gradient estimates compared to pure policy gradient methods like REINFORCE.

### 3. Proximal Policy Optimization (PPO)
- **Overview**: An advanced policy gradient method that improves training stability by preventing large, destructive updates to the policy.
- **Strengths**: Balances exploration and exploitation more effectively, making it a state-of-the-art method for many reinforcement learning tasks.

### 4. Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- **Overview**: An evolutionary algorithm that is particularly effective for optimizing non-linear or non-convex functions. In reinforcement learning, it optimizes policy parameters by sampling from a multivariate normal distribution and adapting its covariance matrix.
- **Strengths**: Does not rely on gradient information, making it robust for optimizing complex, noisy, or non-differentiable objectives.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the necessary libraries. Install the dependencies using:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Dinu23/Policy-based-RL.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Policy-based-RL
   ```

### Running the Algorithms

To train a specific policy-based algorithm on the `Catch` environment:

- **REINFORCE**:
  ```bash
  python monte_carlo_policy_gradeint.py
  ```
- **A2C**:
  ```bash
  python actor_critic.py
  ```
- **PPO**:
  ```bash
  python ppo.py
  ```
- **CMA-ES**:
  ```bash
  python CMAES.py
  ```

## Contact

For questions, issues, or any other inquiries, please reach out to:

- **Name**: Dinu Catalin-Viorel
- **Email**: viorel.dinu00@gmail.com
