# Applications of Reinforcement Learning

## Introduction

This document explores various applications of reinforcement learning across different domains, including game playing, robotics, resource management, and more.

## Game Playing

### 1. Atari Games
```python
class AtariAgent:
    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon: float
    ):
        """
        Initialize Atari agent.

        Args:
            state_dim: State dimensions (height, width, channels)
            action_dim: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            state_dim=state_dim,
            action_dim=1
        )

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Preprocess state for network input.

        Args:
            state: Raw state

        Returns:
            Preprocessed state
        """
        # Convert to grayscale
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        
        # Resize
        state = cv2.resize(state, (84, 84))
        
        # Normalize
        state = state / 255.0
        
        return torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

    def select_action(
        self,
        state: np.ndarray,
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
        state = self.preprocess_state(state)
        
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
            next_q_values = self.target_network(next_states)
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

### 2. Chess
```python
class ChessAgent:
    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        temperature: float
    ):
        """
        Initialize chess agent.

        Args:
            learning_rate: Learning rate
            gamma: Discount factor
            temperature: Temperature for action selection
        """
        self.gamma = gamma
        self.temperature = temperature
        
        # Initialize policy and value networks
        self.policy_network = ChessPolicyNetwork()
        self.value_network = ChessValueNetwork()
        
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate
        )

    def encode_state(self, board: chess.Board) -> torch.Tensor:
        """
        Encode chess board state.

        Args:
            board: Chess board

        Returns:
            Encoded state
        """
        # Create state tensor
        state = torch.zeros(8, 8, 12)
        
        # Encode pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = square // 8
                file = square % 8
                piece_idx = piece.piece_type - 1 + (6 if piece.color else 0)
                state[rank, file, piece_idx] = 1
        
        return state.unsqueeze(0)

    def select_action(
        self,
        board: chess.Board,
        training: bool = True
    ) -> chess.Move:
        """
        Select move using policy network.

        Args:
            board: Chess board
            training: Whether in training mode

        Returns:
            Selected move
        """
        state = self.encode_state(board)
        
        with torch.no_grad():
            move_probs = self.policy_network(state)
            
            if training:
                # Apply temperature
                move_probs = (move_probs / self.temperature).softmax(dim=1)
                move_idx = torch.multinomial(move_probs, 1).item()
            else:
                move_idx = move_probs.argmax().item()
        
        # Convert to chess move
        legal_moves = list(board.legal_moves)
        return legal_moves[move_idx]

    def update(
        self,
        states: List[torch.Tensor],
        moves: List[chess.Move],
        rewards: List[float]
    ) -> Tuple[float, float]:
        """
        Update policy and value networks.

        Args:
            states: List of states
            moves: List of moves
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
        for state, move, R in zip(states, moves, returns):
            move_probs = self.policy_network(state)
            move_idx = list(move.legal_moves).index(move)
            policy_loss -= torch.log(move_probs[0, move_idx]) * R
        
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

## Robotics

### 1. Robot Control
```python
class RobotController:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        tau: float
    ):
        """
        Initialize robot controller.

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

### 2. Robotic Manipulation
```python
class ManipulationAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon: float
    ):
        """
        Initialize manipulation agent.

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
    ) -> torch.Tensor:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return torch.rand(self.action_dim) * 2 - 1
        
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

### 1. Inventory Control
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

### 2. Energy Management
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

## Next Steps

1. Explore [Research Directions](09-research.md)
2. Study [Case Studies](10-case-studies.md)
3. Learn about [Future Developments](11-future.md) 