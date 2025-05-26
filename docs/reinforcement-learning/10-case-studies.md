# Case Studies in Reinforcement Learning

## Introduction

This document presents real-world case studies of reinforcement learning applications, including success stories, challenges faced, and lessons learned.

## Game Playing

### 1. AlphaGo
```python
class AlphaGoAgent:
    def __init__(
        self,
        board_size: int,
        learning_rate: float,
        gamma: float,
        temperature: float
    ):
        """
        Initialize AlphaGo agent.

        Args:
            board_size: Board size
            learning_rate: Learning rate
            gamma: Discount factor
            temperature: Temperature for action selection
        """
        self.board_size = board_size
        self.gamma = gamma
        self.temperature = temperature
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(board_size)
        
        # Initialize value network
        self.value_network = ValueNetwork(board_size)
        
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate
        )

    def select_action(
        self,
        board: np.ndarray,
        training: bool = True
    ) -> Tuple[int, int]:
        """
        Select move using policy network and MCTS.

        Args:
            board: Current board state
            training: Whether in training mode

        Returns:
            Selected move (row, col)
        """
        # Get policy probabilities
        with torch.no_grad():
            policy_probs = self.policy_network(board)
        
        if training:
            # Apply temperature
            policy_probs = (policy_probs / self.temperature).softmax(dim=-1)
            move_idx = torch.multinomial(policy_probs, 1).item()
        else:
            move_idx = policy_probs.argmax().item()
        
        # Convert to board coordinates
        row = move_idx // self.board_size
        col = move_idx % self.board_size
        
        return row, col

    def update(
        self,
        states: List[np.ndarray],
        actions: List[Tuple[int, int]],
        rewards: List[float]
    ) -> Tuple[float, float]:
        """
        Update policy and value networks.

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards

        Returns:
            Tuple of (policy loss, value loss)
        """
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Update policy network
        policy_loss = 0
        for state, action, R in zip(states, actions, returns):
            policy_probs = self.policy_network(state)
            action_idx = action[0] * self.board_size + action[1]
            policy_loss -= torch.log(policy_probs[action_idx]) * R
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        value_loss = 0
        for state, R in zip(states, returns):
            value = self.value_network(state)
            value_loss += F.mse_loss(value, R)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
```

### 2. Dota 2
```python
class Dota2Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon: float
    ):
        """
        Initialize Dota 2 agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        
        # Initialize value network
        self.value_network = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate
        )

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = True
    ) -> int:
        """
        Select action using policy network.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            action_probs = self.policy_network(state)
            return action_probs.argmax().item()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update policy and value networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards

        Returns:
            Tuple of (policy loss, value loss)
        """
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Update policy network
        action_probs = self.policy_network(states)
        action_log_probs = torch.log(action_probs.gather(1, actions))
        policy_loss = -(action_log_probs * returns).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        values = self.value_network(states)
        value_loss = F.mse_loss(values, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
```

## Robotics

### 1. Robotic Manipulation
```python
class ManipulationAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        tau: float
    ):
        """
        Initialize manipulation agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Target network update rate
        """
        self.gamma = gamma
        self.tau = tau
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=learning_rate
        )

    def select_action(
        self,
        state: torch.Tensor,
        noise: float = 0.0
    ) -> torch.Tensor:
        """
        Select action using actor network.

        Args:
            state: Current state
            noise: Action noise

        Returns:
            Selected action
        """
        with torch.no_grad():
            action = self.actor(state)
            if noise > 0:
                action += torch.randn_like(action) * noise
                action = torch.clamp(action, -1, 1)
        return action

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update actor and critic networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Tuple of (actor loss, critic loss)
        """
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_targets()
        
        return actor_loss.item(), critic_loss.item()

    def _update_targets(self):
        """Update target networks."""
        for target_param, param in zip(
            self.target_actor.parameters(),
            self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )
        
        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )
```

### 2. Autonomous Driving
```python
class DrivingAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon: float
    ):
        """
        Initialize driving agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-network
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            state_dim=state_dim,
            action_dim=action_dim
        )

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self, batch_size: int) -> float:
        """
        Update Q-network.

        Args:
            batch_size: Batch size

        Returns:
            Loss value
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions)
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Resource Management

### 1. Energy Management
```python
class EnergyManager:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon: float
    ):
        """
        Initialize energy manager.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-network
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=10000,
            state_dim=state_dim,
            action_dim=action_dim
        )

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self, batch_size: int) -> float:
        """
        Update Q-network.

        Args:
            batch_size: Batch size

        Returns:
            Loss value
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions)
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 2. Inventory Control
```python
class InventoryController:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon: float
    ):
        """
        Initialize inventory controller.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-network
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=10000,
            state_dim=state_dim,
            action_dim=action_dim
        )

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self, batch_size: int) -> float:
        """
        Update Q-network.

        Args:
            batch_size: Batch size

        Returns:
            Loss value
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions)
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Next Steps

1. Learn about [Future Developments](11-future.md)
2. Explore [Implementation Guidelines](07-implementation.md)
3. Study [Research Directions](09-research.md) 