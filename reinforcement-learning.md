# Reinforcement Learning

## Core Concepts

### 1. Markov Decision Process (MDP)

A Markov Decision Process is the mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.

```python
class MDP:
    def __init__(
        self,
        states: Set[State],
        actions: Set[Action],
        transitions: Dict[Tuple[State, Action], Dict[State, float]],
        rewards: Dict[Tuple[State, Action, State], float],
        discount_factor: float
    ):
        """
        Initialize an MDP.

        Args:
            states: Set of possible states
            actions: Set of possible actions
            transitions: Transition probabilities P(s'|s,a)
            rewards: Reward function R(s,a,s')
            discount_factor: Discount factor γ
        """
        pass
```

### 2. Value Functions

#### State-Value Function V(s)
The expected return when starting in state s and following policy π:
```python
def state_value(state: State, policy: Policy) -> float:
    """
    Calculate state-value for a given state and policy.

    Args:
        state: Current state
        policy: Policy to follow

    Returns:
        Expected return from state
    """
    pass
```

#### Action-Value Function Q(s,a)
The expected return when taking action a in state s and following policy π:
```python
def action_value(state: State, action: Action, policy: Policy) -> float:
    """
    Calculate action-value for a given state-action pair and policy.

    Args:
        state: Current state
        action: Action to take
        policy: Policy to follow

    Returns:
        Expected return from state-action pair
    """
    pass
```

## Algorithms

### 1. Value-Based Methods

#### Q-Learning
```python
class QLearning:
    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float
    ):
        """
        Initialize Q-Learning algorithm.

        Args:
            learning_rate: Learning rate α
            discount_factor: Discount factor γ
            exploration_rate: Exploration rate ε
        """
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State
    ) -> None:
        """
        Update Q-values using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table.get((state, action), 0.0)
        next_max_q = max(
            self.q_table.get((next_state, a), 0.0)
            for a in self.actions
        )
        new_q = current_q + self.alpha * (
            reward + self.gamma * next_max_q - current_q
        )
        self.q_table[(state, action)] = new_q
```

#### Deep Q-Network (DQN)
```python
class DQN:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        discount_factor: float
    ):
        """
        Initialize Deep Q-Network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            discount_factor: Discount factor
        """
        self.q_network = self._build_network(
            state_dim,
            action_dim,
            learning_rate
        )
        self.target_network = self._build_network(
            state_dim,
            action_dim,
            learning_rate
        )
        self.memory = ReplayBuffer(10000)
        self.gamma = discount_factor

    def _build_network(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ) -> tf.keras.Model:
        """Build neural network for Q-value approximation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_dim)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mse'
        )
        return model
```

### 2. Policy-Based Methods

#### Policy Gradient
```python
class PolicyGradient:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ):
        """
        Initialize Policy Gradient algorithm.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
        """
        self.policy_network = self._build_network(
            state_dim,
            action_dim,
            learning_rate
        )

    def _build_network(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ) -> tf.keras.Model:
        """Build neural network for policy approximation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=self._policy_gradient_loss
        )
        return model
```

#### Proximal Policy Optimization (PPO)
```python
class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        clip_ratio: float
    ):
        """
        Initialize PPO algorithm.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            clip_ratio: PPO clip ratio
        """
        self.policy_network = self._build_network(
            state_dim,
            action_dim,
            learning_rate
        )
        self.value_network = self._build_value_network(
            state_dim,
            learning_rate
        )
        self.clip_ratio = clip_ratio

    def _build_network(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ) -> tf.keras.Model:
        """Build neural network for policy approximation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=self._ppo_loss
        )
        return model
```

### 3. Actor-Critic Methods

#### Advantage Actor-Critic (A2C)
```python
class A2C:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        discount_factor: float
    ):
        """
        Initialize A2C algorithm.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            discount_factor: Discount factor
        """
        self.actor = self._build_actor(
            state_dim,
            action_dim,
            learning_rate
        )
        self.critic = self._build_critic(
            state_dim,
            learning_rate
        )
        self.gamma = discount_factor

    def _build_actor(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ) -> tf.keras.Model:
        """Build actor network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=self._actor_loss
        )
        return model
```

## Advanced Topics

### 1. Multi-Agent Reinforcement Learning

#### Independent Q-Learning
```python
class IndependentQLearning:
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ):
        """
        Initialize Independent Q-Learning for multiple agents.

        Args:
            num_agents: Number of agents
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
        """
        self.agents = [
            QLearning(state_dim, action_dim, learning_rate)
            for _ in range(num_agents)
        ]
```

#### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
```python
class MADDPG:
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ):
        """
        Initialize MADDPG algorithm.

        Args:
            num_agents: Number of agents
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
        """
        self.actors = [
            self._build_actor(state_dim, action_dim, learning_rate)
            for _ in range(num_agents)
        ]
        self.critics = [
            self._build_critic(
                state_dim * num_agents,
                action_dim * num_agents,
                learning_rate
            )
            for _ in range(num_agents)
        ]
```

### 2. Hierarchical Reinforcement Learning

#### Options Framework
```python
class Option:
    def __init__(
        self,
        name: str,
        policy: Policy,
        termination_condition: Callable[[State], bool],
        initiation_set: Set[State]
    ):
        """
        Initialize an option.

        Args:
            name: Option name
            policy: Option policy
            termination_condition: Termination condition
            initiation_set: Set of states where option can be initiated
        """
        self.name = name
        self.policy = policy
        self.termination_condition = termination_condition
        self.initiation_set = initiation_set
```

#### MAXQ Value Function Decomposition
```python
class MAXQ:
    def __init__(
        self,
        root_task: Task,
        learning_rate: float,
        discount_factor: float
    ):
        """
        Initialize MAXQ algorithm.

        Args:
            root_task: Root task in hierarchy
            learning_rate: Learning rate
            discount_factor: Discount factor
        """
        self.root_task = root_task
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.value_tables = {}
```

### 3. Inverse Reinforcement Learning

#### Maximum Entropy IRL
```python
class MaxEntIRL:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float
    ):
        """
        Initialize Maximum Entropy IRL.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
        """
        self.reward_network = self._build_reward_network(
            state_dim,
            learning_rate
        )
        self.policy_network = self._build_policy_network(
            state_dim,
            action_dim,
            learning_rate
        )
```

## Implementation Guidelines

### 1. Environment Design

```python
class RLEnvironment:
    def __init__(
        self,
        state_space: Space,
        action_space: Space,
        reward_function: Callable[[State, Action, State], float]
    ):
        """
        Initialize RL environment.

        Args:
            state_space: State space definition
            action_space: Action space definition
            reward_function: Reward function
        """
        self.state_space = state_space
        self.action_space = action_space
        self.reward_function = reward_function

    def reset(self) -> State:
        """Reset environment to initial state."""
        pass

    def step(self, action: Action) -> Tuple[State, float, bool, dict]:
        """
        Execute action in environment.

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
```

### 2. Training Process

```python
class RLTraining:
    def __init__(
        self,
        agent: RLAgent,
        environment: RLEnvironment,
        num_episodes: int,
        max_steps: int
    ):
        """
        Initialize RL training process.

        Args:
            agent: RL agent
            environment: RL environment
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
        """
        self.agent = agent
        self.environment = environment
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def train(self) -> Dict[str, List[float]]:
        """
        Train the agent.

        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': []
        }

        for episode in range(self.num_episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(self.max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.environment.step(action)
                
                self.agent.update(state, action, reward, next_state)
                
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)

        return metrics
```

### 3. Evaluation and Monitoring

```python
class RLEvaluation:
    def __init__(
        self,
        agent: RLAgent,
        environment: RLEnvironment,
        num_episodes: int
    ):
        """
        Initialize RL evaluation process.

        Args:
            agent: Trained RL agent
            environment: RL environment
            num_episodes: Number of evaluation episodes
        """
        self.agent = agent
        self.environment = environment
        self.num_episodes = num_episodes

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the agent.

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'mean_reward': 0.0,
            'mean_length': 0.0,
            'success_rate': 0.0
        }

        for episode in range(self.num_episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_length = 0
            success = False

            while True:
                action = self.agent.select_action(state, evaluation=True)
                next_state, reward, done, info = self.environment.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    success = info.get('success', False)
                    break

            metrics['mean_reward'] += episode_reward
            metrics['mean_length'] += episode_length
            metrics['success_rate'] += int(success)

        # Calculate averages
        metrics['mean_reward'] /= self.num_episodes
        metrics['mean_length'] /= self.num_episodes
        metrics['success_rate'] /= self.num_episodes

        return metrics
```

## Applications

### 1. Game Playing

```python
class GameEnvironment(RLEnvironment):
    def __init__(
        self,
        game_type: str,
        board_size: Tuple[int, int]
    ):
        """
        Initialize game environment.

        Args:
            game_type: Type of game (e.g., 'chess', 'go')
            board_size: Size of game board
        """
        super().__init__(
            state_space=self._create_state_space(board_size),
            action_space=self._create_action_space(game_type),
            reward_function=self._create_reward_function()
        )
```

### 2. Robotics

```python
class RobotEnvironment(RLEnvironment):
    def __init__(
        self,
        robot_type: str,
        task_type: str
    ):
        """
        Initialize robot environment.

        Args:
            robot_type: Type of robot
            task_type: Type of task
        """
        super().__init__(
            state_space=self._create_state_space(robot_type),
            action_space=self._create_action_space(robot_type),
            reward_function=self._create_reward_function(task_type)
        )
```

### 3. Resource Management

```python
class ResourceEnvironment(RLEnvironment):
    def __init__(
        self,
        resource_types: List[str],
        num_resources: int
    ):
        """
        Initialize resource management environment.

        Args:
            resource_types: Types of resources
            num_resources: Number of resources
        """
        super().__init__(
            state_space=self._create_state_space(resource_types),
            action_space=self._create_action_space(num_resources),
            reward_function=self._create_reward_function()
        )
```

## Best Practices

### 1. Hyperparameter Tuning

```python
class HyperparameterTuner:
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        evaluation_metric: str
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            param_grid: Grid of hyperparameters
            evaluation_metric: Metric to optimize
        """
        self.param_grid = param_grid
        self.evaluation_metric = evaluation_metric

    def tune(
        self,
        agent_class: Type[RLAgent],
        environment: RLEnvironment
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters.

        Returns:
            Best hyperparameters
        """
        best_params = None
        best_score = float('-inf')

        for params in self._generate_param_combinations():
            agent = agent_class(**params)
            score = self._evaluate_agent(agent, environment)
            
            if score > best_score:
                best_score = score
                best_params = params

        return best_params
```

### 2. Experiment Tracking

```python
class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.mlflow = mlflow

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int
    ) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        for name, value in metrics.items():
            self.mlflow.log_metric(name, value, step=step)
```

### 3. Model Deployment

```python
class ModelDeployer:
    def __init__(
        self,
        model_path: str,
        deployment_config: Dict[str, Any]
    ):
        """
        Initialize model deployer.

        Args:
            model_path: Path to trained model
            deployment_config: Deployment configuration
        """
        self.model_path = model_path
        self.deployment_config = deployment_config

    def deploy(self) -> str:
        """
        Deploy model.

        Returns:
            Deployment endpoint
        """
        # Load model
        model = self._load_model()
        
        # Prepare deployment
        deployment = self._prepare_deployment(model)
        
        # Deploy
        endpoint = self._deploy_model(deployment)
        
        return endpoint
``` 