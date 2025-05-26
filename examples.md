# Examples

## Basic Examples

### 1. Simple Agent

```python
from agents.core import Agent
from agents.environment import Environment

# Create environment
env = Environment(
    size=(100, 100),
    obstacles=True
)

# Create agent
agent = Agent(
    name="SimpleAgent",
    goals=["explore", "collect"],
    environment=env
)

# Run agent
agent.start()
agent.run(duration=3600)  # Run for 1 hour
```

### 2. Learning Agent

```python
from agents.core import LearningAgent
from agents.environment import Environment
from agents.learning import QLearning

# Create environment
env = Environment(
    size=(50, 50),
    obstacles=True
)

# Create learning agent
agent = LearningAgent(
    name="LearningAgent",
    goals=["navigate", "collect"],
    environment=env,
    learning_algorithm=QLearning(
        learning_rate=0.1,
        discount_factor=0.9
    )
)

# Train agent
agent.train(episodes=1000)

# Run agent
agent.start()
```

## Advanced Examples

### 1. Multi-Agent System

```python
from agents.core import Agent, MultiAgentSystem
from agents.environment import Environment
from agents.communication import MessageBus

# Create environment
env = Environment(
    size=(200, 200),
    obstacles=True
)

# Create message bus
message_bus = MessageBus()

# Create agents
explorer = Agent(
    name="Explorer",
    goals=["explore", "map"],
    environment=env
)

collector = Agent(
    name="Collector",
    goals=["collect", "return"],
    environment=env
)

# Create multi-agent system
system = MultiAgentSystem(
    agents=[explorer, collector],
    message_bus=message_bus,
    environment=env
)

# Run system
system.start()
```

### 2. Custom Agent

```python
from agents.core import BaseAgent
from agents.actions import Action
from agents.perception import Sensor

class CustomAgent(BaseAgent):
    def __init__(self, name, goals, environment):
        super().__init__(name, goals, environment)
        self.sensors = [
            Sensor("vision", range=10),
            Sensor("proximity", range=5)
        ]
        self.actions = [
            Action("move", cost=1),
            Action("collect", cost=2),
            Action("analyze", cost=3)
        ]

    def perceive(self):
        observations = {}
        for sensor in self.sensors:
            observations[sensor.name] = sensor.observe()
        return observations

    def decide(self, observations):
        # Custom decision making logic
        if observations["proximity"] < 2:
            return self.actions[0]  # Move
        elif observations["vision"].has_resource:
            return self.actions[1]  # Collect
        return self.actions[2]  # Analyze

# Usage
agent = CustomAgent(
    name="CustomAgent",
    goals=["explore", "collect", "analyze"],
    environment=Environment()
)
```

## Use Cases

### 1. Resource Collection

```python
from agents.core import Agent
from agents.environment import Environment
from agents.actions import Action
from agents.goals import Goal

# Define custom actions
class CollectAction(Action):
    def execute(self, agent, target):
        if target.has_resource:
            resource = target.collect()
            agent.inventory.add(resource)
            return True
        return False

# Define custom goal
class CollectionGoal(Goal):
    def __init__(self, target_amount):
        super().__init__("collect")
        self.target_amount = target_amount

    def is_achieved(self, agent):
        return agent.inventory.total >= self.target_amount

# Create agent
agent = Agent(
    name="Collector",
    goals=[CollectionGoal(100)],
    actions=[CollectAction()],
    environment=Environment()
)
```

### 2. Path Planning

```python
from agents.core import Agent
from agents.environment import Environment
from agents.planning import AStarPlanner

# Create environment with obstacles
env = Environment(
    size=(100, 100),
    obstacles=True
)

# Create planner
planner = AStarPlanner(
    heuristic="manhattan",
    diagonal_movement=True
)

# Create agent
agent = Agent(
    name="PathPlanner",
    goals=["navigate"],
    environment=env,
    planner=planner
)

# Plan path
start = (0, 0)
goal = (99, 99)
path = agent.plan_path(start, goal)
```

### 3. Learning from Experience

```python
from agents.core import LearningAgent
from agents.learning import ReinforcementLearning
from agents.environment import Environment

# Create environment
env = Environment(
    size=(50, 50),
    obstacles=True
)

# Create learning algorithm
learning = ReinforcementLearning(
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=0.2
)

# Create agent
agent = LearningAgent(
    name="Learner",
    goals=["navigate", "collect"],
    environment=env,
    learning_algorithm=learning
)

# Train agent
agent.train(
    episodes=1000,
    max_steps=100,
    render=True
)

# Save learned policy
agent.save_policy("learned_policy.pkl")
```

## Integration Examples

### 1. Web API Integration

```python
from agents.core import Agent
from agents.environment import WebEnvironment
from agents.actions import HTTPAction

# Create web environment
env = WebEnvironment(
    base_url="https://api.example.com",
    auth_token="your-token"
)

# Create HTTP actions
actions = [
    HTTPAction("GET", "/data"),
    HTTPAction("POST", "/process"),
    HTTPAction("PUT", "/update")
]

# Create agent
agent = Agent(
    name="WebAgent",
    goals=["fetch", "process", "update"],
    environment=env,
    actions=actions
)
```

### 2. Database Integration

```python
from agents.core import Agent
from agents.environment import DatabaseEnvironment
from agents.actions import DatabaseAction

# Create database environment
env = DatabaseEnvironment(
    connection_string="postgresql://user:pass@localhost/db",
    pool_size=5
)

# Create database actions
actions = [
    DatabaseAction("SELECT", "users"),
    DatabaseAction("INSERT", "logs"),
    DatabaseAction("UPDATE", "status")
]

# Create agent
agent = Agent(
    name="DatabaseAgent",
    goals=["monitor", "maintain", "optimize"],
    environment=env,
    actions=actions
)
```

## Best Practices

1. **Error Handling**
```python
try:
    agent.start()
except AgentError as e:
    logger.error(f"Agent failed to start: {e}")
    # Handle error
finally:
    agent.cleanup()
```

2. **Resource Management**
```python
with Agent(name="ResourceAgent") as agent:
    agent.start()
    agent.run(duration=3600)
```

3. **Configuration Management**
```python
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

agent = Agent(**config["agent"])
```

4. **Logging**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Starting agent")
``` 