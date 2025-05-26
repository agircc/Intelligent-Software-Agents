# Implementation Guidelines

## Introduction

This document provides comprehensive guidelines for implementing reinforcement learning algorithms, including environment design, training procedures, evaluation methods, and best practices.

## Environment Design

### 1. Environment Interface
```python
class Environment:
    def __init__(self):
        """Initialize environment."""
        pass

    def reset(self) -> State:
        """
        Reset environment to initial state.

        Returns:
            Initial state
        """
        pass

    def step(
        self,
        action: Action
    ) -> Tuple[State, float, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next state, reward, done, info)
        """
        pass

    def render(self) -> None:
        """Render environment."""
        pass

    def close(self) -> None:
        """Close environment."""
        pass
```

### 2. State and Action Spaces
```python
class StateSpace:
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        low: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None
    ):
        """
        Initialize state space.

        Args:
            shape: State shape
            dtype: State data type
            low: Lower bounds
            high: Upper bounds
        """
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high

    def sample(self) -> np.ndarray:
        """
        Sample random state.

        Returns:
            Random state
        """
        if self.low is None or self.high is None:
            return np.random.randn(*self.shape).astype(self.dtype)
        return np.random.uniform(
            self.low,
            self.high,
            self.shape
        ).astype(self.dtype)

class ActionSpace:
    def __init__(
        self,
        n: int,
        dtype: np.dtype = np.int64
    ):
        """
        Initialize discrete action space.

        Args:
            n: Number of actions
            dtype: Action data type
        """
        self.n = n
        self.dtype = dtype

    def sample(self) -> int:
        """
        Sample random action.

        Returns:
            Random action
        """
        return np.random.randint(0, self.n, dtype=self.dtype)
```

## Training Process

### 1. Experience Collection
```python
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Buffer capacity
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch of experiences.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
```

### 2. Training Loop
```python
def train(
    agent: Agent,
    env: Environment,
    num_episodes: int,
    max_steps: int,
    batch_size: int,
    update_frequency: int,
    target_update_frequency: int,
    save_frequency: int,
    log_frequency: int
) -> Dict[str, List[float]]:
    """
    Train agent.

    Args:
        agent: Agent to train
        env: Environment
        num_episodes: Number of episodes
        max_steps: Maximum steps per episode
        batch_size: Batch size
        update_frequency: Policy update frequency
        target_update_frequency: Target network update frequency
        save_frequency: Model save frequency
        log_frequency: Logging frequency

    Returns:
        Dictionary of training metrics
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )
            
            # Update agent
            if len(agent.replay_buffer) >= batch_size:
                if step % update_frequency == 0:
                    loss = agent.update(batch_size)
                    metrics['losses'].append(loss)
                
                if step % target_update_frequency == 0:
                    agent.update_target_network()
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # Log metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        
        if episode % log_frequency == 0:
            print(f"Episode {episode}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Length: {episode_length}")
            print(f"Average Loss: {np.mean(metrics['losses'][-100:]):.4f}")
            print()
        
        # Save model
        if episode % save_frequency == 0:
            agent.save(f"checkpoints/agent_{episode}.pt")
    
    return metrics
```

## Evaluation Methods

### 1. Performance Metrics
```python
def evaluate_agent(
    agent: Agent,
    env: Environment,
    num_episodes: int,
    max_steps: int,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate agent performance.

    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Number of episodes
        max_steps: Maximum steps per episode
        render: Whether to render environment

    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
```

### 2. Visualization Tools
```python
def plot_training_curves(
    metrics: Dict[str, List[float]],
    window_size: int = 100
) -> None:
    """
    Plot training curves.

    Args:
        metrics: Dictionary of training metrics
        window_size: Window size for moving average
    """
    plt.figure(figsize=(12, 4))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    rewards = metrics['episode_rewards']
    plt.plot(rewards, alpha=0.3)
    plt.plot(
        np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode='valid'
        ),
        label=f'{window_size}-episode moving average'
    )
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    
    # Plot losses
    plt.subplot(1, 2, 2)
    losses = metrics['losses']
    plt.plot(losses, alpha=0.3)
    plt.plot(
        np.convolve(
            losses,
            np.ones(window_size) / window_size,
            mode='valid'
        ),
        label=f'{window_size}-update moving average'
    )
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## Best Practices

### 1. Hyperparameter Tuning
```python
def tune_hyperparameters(
    agent_class: Type[Agent],
    env: Environment,
    param_grid: Dict[str, List[Any]],
    num_episodes: int,
    num_trials: int
) -> Dict[str, Any]:
    """
    Tune hyperparameters using random search.

    Args:
        agent_class: Agent class
        env: Environment
        param_grid: Parameter grid
        num_episodes: Number of episodes per trial
        num_trials: Number of trials

    Returns:
        Best hyperparameters
    """
    best_reward = float('-inf')
    best_params = None
    
    for trial in range(num_trials):
        # Sample hyperparameters
        params = {
            param: random.choice(values)
            for param, values in param_grid.items()
        }
        
        # Train agent
        agent = agent_class(**params)
        metrics = train(agent, env, num_episodes)
        mean_reward = np.mean(metrics['episode_rewards'][-100:])
        
        # Update best parameters
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params
    
    return best_params
```

### 2. Experiment Tracking
```python
class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        params: Dict[str, Any]
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of experiment
            params: Experiment parameters
        """
        self.experiment_name = experiment_name
        self.params = params
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # Create experiment directory
        self.log_dir = f"experiments/{experiment_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save parameters
        with open(f"{self.log_dir}/params.json", 'w') as f:
            json.dump(params, f, indent=4)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        for name, value in metrics.items():
            self.metrics[name].append((step, value))
        
        # Save metrics
        with open(f"{self.log_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=4)

    def save_model(
        self,
        model: nn.Module,
        step: int
    ) -> None:
        """
        Save model.

        Args:
            model: Model to save
            step: Current step
        """
        torch.save(
            model.state_dict(),
            f"{self.log_dir}/model_{step}.pt"
        )

    def load_model(
        self,
        model: nn.Module,
        step: int
    ) -> None:
        """
        Load model.

        Args:
            model: Model to load
            step: Step to load
        """
        model.load_state_dict(
            torch.load(f"{self.log_dir}/model_{step}.pt")
        )
```

## Next Steps

1. Learn about [Applications](08-applications.md)
2. Explore [Research Directions](09-research.md)
3. Study [Case Studies](10-case-studies.md) 