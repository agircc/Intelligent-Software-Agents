# API Reference

## Core Components

### Agent

```python
class Agent:
    def __init__(
        self,
        name: str,
        goals: List[str],
        environment: Environment,
        actions: Optional[List[Action]] = None,
        planner: Optional[Planner] = None
    ):
        """
        Initialize an agent.

        Args:
            name: Agent identifier
            goals: List of goal names
            environment: Environment instance
            actions: Optional list of actions
            planner: Optional planner instance
        """
        pass

    def start(self) -> None:
        """Start the agent."""
        pass

    def stop(self) -> None:
        """Stop the agent."""
        pass

    def run(self, duration: Optional[int] = None) -> None:
        """
        Run the agent.

        Args:
            duration: Optional duration in seconds
        """
        pass

    def perceive(self) -> Dict[str, Any]:
        """
        Perceive the environment.

        Returns:
            Dictionary of observations
        """
        pass

    def decide(self, observations: Dict[str, Any]) -> Action:
        """
        Make a decision based on observations.

        Args:
            observations: Dictionary of observations

        Returns:
            Selected action
        """
        pass

    def act(self, action: Action) -> bool:
        """
        Execute an action.

        Args:
            action: Action to execute

        Returns:
            Success status
        """
        pass
```

### Environment

```python
class Environment:
    def __init__(
        self,
        size: Tuple[int, int],
        obstacles: bool = False,
        resources: bool = False
    ):
        """
        Initialize an environment.

        Args:
            size: Environment dimensions
            obstacles: Whether to include obstacles
            resources: Whether to include resources
        """
        pass

    def reset(self) -> None:
        """Reset the environment."""
        pass

    def step(self, action: Action) -> Tuple[Dict[str, Any], float, bool]:
        """
        Execute an environment step.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observations, reward, done)
        """
        pass

    def render(self) -> None:
        """Render the environment."""
        pass
```

### Action

```python
class Action:
    def __init__(
        self,
        name: str,
        cost: float = 1.0,
        preconditions: Optional[List[str]] = None,
        effects: Optional[List[str]] = None
    ):
        """
        Initialize an action.

        Args:
            name: Action identifier
            cost: Action cost
            preconditions: List of preconditions
            effects: List of effects
        """
        pass

    def execute(self, agent: Agent, **kwargs) -> bool:
        """
        Execute the action.

        Args:
            agent: Agent instance
            **kwargs: Additional arguments

        Returns:
            Success status
        """
        pass
```

## Learning Components

### LearningAgent

```python
class LearningAgent(Agent):
    def __init__(
        self,
        name: str,
        goals: List[str],
        environment: Environment,
        learning_algorithm: LearningAlgorithm
    ):
        """
        Initialize a learning agent.

        Args:
            name: Agent identifier
            goals: List of goal names
            environment: Environment instance
            learning_algorithm: Learning algorithm instance
        """
        pass

    def train(
        self,
        episodes: int,
        max_steps: Optional[int] = None,
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train the agent.

        Args:
            episodes: Number of episodes
            max_steps: Maximum steps per episode
            render: Whether to render episodes

        Returns:
            Training metrics
        """
        pass

    def save_policy(self, path: str) -> None:
        """
        Save the learned policy.

        Args:
            path: Path to save policy
        """
        pass

    def load_policy(self, path: str) -> None:
        """
        Load a policy.

        Args:
            path: Path to policy file
        """
        pass
```

### LearningAlgorithm

```python
class LearningAlgorithm:
    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float
    ):
        """
        Initialize a learning algorithm.

        Args:
            learning_rate: Learning rate
            discount_factor: Discount factor
            exploration_rate: Exploration rate
        """
        pass

    def update(
        self,
        state: Any,
        action: Action,
        reward: float,
        next_state: Any
    ) -> None:
        """
        Update the learning algorithm.

        Args:
            state: Current state
            action: Executed action
            reward: Received reward
            next_state: Next state
        """
        pass

    def select_action(self, state: Any) -> Action:
        """
        Select an action.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        pass
```

## Planning Components

### Planner

```python
class Planner:
    def __init__(
        self,
        heuristic: str = "manhattan",
        diagonal_movement: bool = False
    ):
        """
        Initialize a planner.

        Args:
            heuristic: Heuristic function
            diagonal_movement: Whether to allow diagonal movement
        """
        pass

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        environment: Environment
    ) -> List[Tuple[int, int]]:
        """
        Plan a path.

        Args:
            start: Start position
            goal: Goal position
            environment: Environment instance

        Returns:
            List of positions
        """
        pass
```

## Communication Components

### MessageBus

```python
class MessageBus:
    def __init__(self):
        """Initialize a message bus."""
        pass

    def publish(self, topic: str, message: Any) -> None:
        """
        Publish a message.

        Args:
            topic: Message topic
            message: Message content
        """
        pass

    def subscribe(self, topic: str, callback: Callable) -> None:
        """
        Subscribe to a topic.

        Args:
            topic: Message topic
            callback: Callback function
        """
        pass
```

### MultiAgentSystem

```python
class MultiAgentSystem:
    def __init__(
        self,
        agents: List[Agent],
        message_bus: MessageBus,
        environment: Environment
    ):
        """
        Initialize a multi-agent system.

        Args:
            agents: List of agents
            message_bus: Message bus instance
            environment: Environment instance
        """
        pass

    def start(self) -> None:
        """Start the system."""
        pass

    def stop(self) -> None:
        """Stop the system."""
        pass

    def run(self, duration: Optional[int] = None) -> None:
        """
        Run the system.

        Args:
            duration: Optional duration in seconds
        """
        pass
```

## Utility Components

### Logger

```python
class Logger:
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        format: str = None
    ):
        """
        Initialize a logger.

        Args:
            name: Logger name
            level: Logging level
            format: Log format
        """
        pass

    def info(self, message: str) -> None:
        """
        Log info message.

        Args:
            message: Log message
        """
        pass

    def error(self, message: str) -> None:
        """
        Log error message.

        Args:
            message: Log message
        """
        pass

    def debug(self, message: str) -> None:
        """
        Log debug message.

        Args:
            message: Log message
        """
        pass
```

### Config

```python
class Config:
    def __init__(self, path: str):
        """
        Initialize configuration.

        Args:
            path: Config file path
        """
        pass

    def load(self) -> Dict[str, Any]:
        """
        Load configuration.

        Returns:
            Configuration dictionary
        """
        pass

    def save(self, config: Dict[str, Any]) -> None:
        """
        Save configuration.

        Args:
            config: Configuration dictionary
        """
        pass
```

## Error Handling

### AgentError

```python
class AgentError(Exception):
    """Base class for agent errors."""
    pass

class ActionError(AgentError):
    """Action execution error."""
    pass

class PlanningError(AgentError):
    """Planning error."""
    pass

class LearningError(AgentError):
    """Learning error."""
    pass
```

## Type Definitions

```python
from typing import List, Dict, Any, Tuple, Optional, Callable

# Basic types
Position = Tuple[int, int]
Observation = Dict[str, Any]
Reward = float
Done = bool

# Component types
AgentState = Dict[str, Any]
ActionResult = Tuple[Observation, Reward, Done]
Policy = Dict[AgentState, Action]
```

## Constants

```python
# Environment
DEFAULT_SIZE = (100, 100)
DEFAULT_OBSTACLES = True
DEFAULT_RESOURCES = True

# Learning
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.9
DEFAULT_EXPLORATION_RATE = 0.2

# Planning
DEFAULT_HEURISTIC = "manhattan"
DEFAULT_DIAGONAL_MOVEMENT = False

# Communication
DEFAULT_MESSAGE_TIMEOUT = 5.0
DEFAULT_RETRY_ATTEMPTS = 3
``` 