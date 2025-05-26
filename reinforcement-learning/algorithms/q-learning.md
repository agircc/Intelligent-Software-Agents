# Q-Learning Algorithm

## 1. Introduction

Q-Learning is a value-based reinforcement learning algorithm that learns an action-value function (Q-function) to find the optimal policy. It is a model-free algorithm, meaning it can learn without knowing the environment's model.

### 1.1 Core Idea

The core idea of Q-Learning is to learn the optimal policy by iteratively updating Q-values. The Q-value represents the expected cumulative reward for taking a specific action in a specific state. The algorithm's goal is to find the policy that maximizes the long-term cumulative reward.

### 1.2 Mathematical Formulation

The Q-Learning update rule is as follows:

\[Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]\]

where:
- \(Q(s_t, a_t)\) is the Q-value for the current state-action pair
- \(\alpha\) is the learning rate
- \(r_t\) is the immediate reward
- \(\gamma\) is the discount factor
- \(\max_a Q(s_{t+1}, a)\) is the maximum Q-value for the next state

## 2. Applications

Q-Learning is suitable for the following scenarios:

1. **Discrete State and Action Spaces**: Q-Learning performs best when states and actions are discrete
2. **Model-Free Environments**: No need to know the environment's transition probabilities
3. **Offline Learning**: Can learn from historical data
4. **Single-Agent Problems**: Suitable for single-agent decision-making problems

### 2.1 Typical Applications

- Game AI (e.g., maze navigation, Snake game)
- Robot Control
- Resource Scheduling
- Recommendation Systems

## 3. Implementation

### 3.1 Basic Implementation

```python
import numpy as np
import random

class QLearning:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01
    ):
        """
        Initialize Q-Learning algorithm
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate
            discount_factor: Discount factor
            exploration_rate: Exploration rate
            exploration_decay: Exploration rate decay
            min_exploration_rate: Minimum exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
    
    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: best action
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-value
        
        Args:
            state: Current state
            action: Executed action
            reward: Received reward
            next_state: Next state
        """
        # Calculate target Q-value
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Update Q-value
        self.q_table[state, action] += self.learning_rate * (
            target - self.q_table[state, action]
        )
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
```

### 3.2 Advanced Implementation (Using Neural Networks)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Initialize Q-network
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            hidden_size: Size of hidden layers
        """
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Q-value tensor
        """
        return self.network(state)

class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10
    ):
        """
        Initialize DQN agent
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate
            discount_factor: Discount factor
            exploration_rate: Exploration rate
            exploration_decay: Exploration rate decay
            min_exploration_rate: Minimum exploration rate
            memory_size: Size of replay buffer
            batch_size: Batch size
            target_update: Target network update frequency
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Initialize Q-network and target network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training steps
        self.steps = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: best action
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool):
        """
        Update Q-network
        
        Args:
            state: Current state
            action: Executed action
            reward: Received reward
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        
        # Skip update if not enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Calculate current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Calculate loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
```

## 4. Example: Maze Navigation

Let's demonstrate Q-Learning with a simple maze navigation problem:

```python
import numpy as np
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, size: int = 5):
        """
        Initialize maze environment
        
        Args:
            size: Maze size
        """
        self.size = size
        self.state_size = size * size
        self.action_size = 4  # up, down, left, right
        
        # Initialize maze
        self.maze = np.zeros((size, size))
        self.maze[size-1, size-1] = 1  # goal position
        
        # Initialize agent position
        self.agent_pos = [0, 0]
    
    def reset(self) -> int:
        """
        Reset environment
        
        Returns:
            Initial state
        """
        self.agent_pos = [0, 0]
        return self._get_state()
    
    def step(self, action: int) -> tuple:
        """
        Execute action
        
        Args:
            action: Action (0: up, 1: down, 2: left, 3: right)
            
        Returns:
            (next state, reward, done)
        """
        # Save current position
        old_pos = self.agent_pos.copy()
        
        # Update position
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        
        # Calculate reward
        if self.agent_pos == [self.size-1, self.size-1]:
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False
        
        return self._get_state(), reward, done
    
    def _get_state(self) -> int:
        """
        Get current state
        
        Returns:
            State index
        """
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def render(self):
        """
        Render maze
        """
        maze = self.maze.copy()
        maze[self.agent_pos[0], self.agent_pos[1]] = 2
        
        plt.imshow(maze, cmap='hot')
        plt.show()

# Train agent
def train_agent(episodes: int = 1000):
    """
    Train agent
    
    Args:
        episodes: Number of training episodes
    """
    # Create environment and agent
    env = Maze()
    agent = QLearning(
        state_size=env.state_size,
        action_size=env.action_size
    )
    
    # Record rewards
    rewards = []
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Update Q-value
            agent.update(state, action, reward, next_state)
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        # Print average reward every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return agent, rewards

# Test agent
def test_agent(agent: QLearning, episodes: int = 10):
    """
    Test agent
    
    Args:
        agent: Trained agent
        episodes: Number of test episodes
    """
    env = Maze()
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            state, _, done = env.step(action)
            
            # Render environment
            env.render()

# Main function
if __name__ == "__main__":
    # Train agent
    agent, rewards = train_agent()
    
    # Plot rewards
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
    
    # Test agent
    test_agent(agent)
```

## 5. Algorithm Analysis

### 5.1 Advantages

1. **Model-Free Learning**: No need to know the environment's model
2. **Offline Learning**: Can learn from historical data
3. **Convergence Guarantee**: Can converge to optimal policy under certain conditions
4. **Simple Implementation**: Both theory and implementation are relatively simple

### 5.2 Disadvantages

1. **State Space Limitation**: Requires discretization for continuous state spaces
2. **Exploration Efficiency**: Simple exploration strategy may lead to inefficient learning
3. **Memory Consumption**: Q-table size grows exponentially with state and action spaces
4. **Convergence Speed**: May converge slowly in some cases

## 6. Improvements

1. **Function Approximation**: Use neural networks to handle continuous state spaces
2. **Prioritized Experience Replay**: Sample experiences based on TD error magnitude
3. **Double Q-Learning**: Use two Q-networks to reduce overestimation
4. **Distributional Q-Learning**: Learn Q-value distribution instead of expected value

## 7. Summary

Q-Learning is one of the most fundamental and important algorithms in reinforcement learning. It learns the optimal policy by iteratively updating Q-values and has the advantages of model-free and offline learning. Although it has some limitations, Q-Learning can be applied to various practical problems through appropriate improvements and extensions.

## 8. References

1. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
3. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning. In AAAI (Vol. 16, pp. 2094-2100). 