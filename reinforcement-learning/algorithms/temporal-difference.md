# Temporal Difference Learning in Reinforcement Learning

## 0. A Concrete Example: Food Delivery Tracking

Imagine you order food delivery through an app. The app initially estimates your food will arrive in 30 minutes. As you wait, you keep checking the app and update your expectation based on new information:

- **At order time**: The app says delivery will take 30 minutes. You expect to eat at 7:00pm.
- **10 minutes later**: The app shows the courier has just picked up your food and now estimates 25 minutes left. You update your expectation: maybe 7:05pm.
- **5 minutes later**: The courier is on the way, but traffic is heavy. The app now says 20 minutes left. You adjust your expectation to 7:10pm.
- **Suddenly, the courier stops moving on the map**: The app adds 5 more minutes to the estimate. You now expect 7:15pm.
- **Finally, the courier enters your neighborhood**: The app says 3 minutes left. You expect to get your food at 7:13pm.

At each step, you revise your prediction of when the food will arrive, based on the latest available information. You don't wait until the food actually arrives to learn; instead, you continuously update your expectations as new data comes in. This is the essence of Temporal Difference learning: learning and adjusting predictions step by step, as new evidence appears, rather than only after the final outcome is known.

## 1. Introduction

Temporal Difference (TD) learning is a class of model-free reinforcement learning algorithms that learn by bootstrapping from the current estimate of the value function. Unlike Monte Carlo methods that wait until the end of an episode, TD methods update estimates based on the difference between temporally successive predictions.

### 1.1 Core Concepts

- **Bootstrapping**: Using current estimates to update other estimates
- **TD Error**: The difference between the current estimate and the target estimate
- **TD(λ)**: A family of algorithms that combine TD and Monte Carlo methods
- **Eligibility Traces**: A mechanism to handle delayed rewards and credit assignment

### 1.2 Mathematical Formulation

The basic TD update rule is:

\[V(s_t) \leftarrow V(s_t) + \alpha[r_t + \gamma V(s_{t+1}) - V(s_t)]\]

where:
- \(V(s_t)\) is the value estimate for state \(s_t\)
- \(\alpha\) is the learning rate
- \(r_t\) is the immediate reward
- \(\gamma\) is the discount factor
- \(V(s_{t+1})\) is the value estimate for the next state

## 2. Basic TD Algorithms

### 2.1 TD(0)

```python
class TD0:
    def __init__(
        self,
        state_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95
    ):
        """
        Initialize TD(0) algorithm
        
        Args:
            state_size: Size of state space
            learning_rate: Learning rate
            discount_factor: Discount factor
        """
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize value function
        self.V = np.zeros(state_size)
    
    def update(self, state: int, next_state: int, reward: float):
        """
        Update value estimates using TD(0)
        
        Args:
            state: Current state
            next_state: Next state
            reward: Immediate reward
        """
        # Calculate TD error
        td_error = reward + self.discount_factor * self.V[next_state] - self.V[state]
        
        # Update value estimate
        self.V[state] += self.learning_rate * td_error
```

### 2.2 SARSA (State-Action-Reward-State-Action)

```python
class SARSA:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        """
        Initialize SARSA algorithm
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate
            discount_factor: Discount factor
            epsilon: Exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize action-value function
        self.Q = np.zeros((state_size, action_size))
    
    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.Q[state])
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int
    ):
        """
        Update action-value estimates using SARSA
        
        Args:
            state: Current state
            action: Current action
            reward: Immediate reward
            next_state: Next state
            next_action: Next action
        """
        # Calculate TD error
        td_error = (reward + 
                   self.discount_factor * self.Q[next_state, next_action] - 
                   self.Q[state, action])
        
        # Update action-value estimate
        self.Q[state, action] += self.learning_rate * td_error
```

### 2.3 Q-Learning

```python
class QLearning:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        """
        Initialize Q-Learning algorithm
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate
            discount_factor: Discount factor
            epsilon: Exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize action-value function
        self.Q = np.zeros((state_size, action_size))
    
    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Update action-value estimates using Q-Learning
        
        Args:
            state: Current state
            action: Current action
            reward: Immediate reward
            next_state: Next state
        """
        # Calculate TD error
        td_error = (reward + 
                   self.discount_factor * np.max(self.Q[next_state]) - 
                   self.Q[state, action])
        
        # Update action-value estimate
        self.Q[state, action] += self.learning_rate * td_error
```

## 3. Advanced TD Methods

### 3.1 TD(λ)

```python
class TDLambda:
    def __init__(
        self,
        state_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        lambda_param: float = 0.9
    ):
        """
        Initialize TD(λ) algorithm
        
        Args:
            state_size: Size of state space
            learning_rate: Learning rate
            discount_factor: Discount factor
            lambda_param: Lambda parameter for eligibility traces
        """
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lambda_param = lambda_param
        
        # Initialize value function
        self.V = np.zeros(state_size)
        
        # Initialize eligibility traces
        self.eligibility = np.zeros(state_size)
    
    def update(self, state: int, next_state: int, reward: float):
        """
        Update value estimates using TD(λ)
        
        Args:
            state: Current state
            next_state: Next state
            reward: Immediate reward
        """
        # Calculate TD error
        td_error = reward + self.discount_factor * self.V[next_state] - self.V[state]
        
        # Update eligibility traces
        self.eligibility *= self.discount_factor * self.lambda_param
        self.eligibility[state] += 1
        
        # Update value estimates
        self.V += self.learning_rate * td_error * self.eligibility
```

### 3.2 Expected SARSA

```python
class ExpectedSARSA:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        """
        Initialize Expected SARSA algorithm
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate
            discount_factor: Discount factor
            epsilon: Exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize action-value function
        self.Q = np.zeros((state_size, action_size))
    
    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Update action-value estimates using Expected SARSA
        
        Args:
            state: Current state
            action: Current action
            reward: Immediate reward
            next_state: Next state
        """
        # Calculate expected value of next state
        policy = np.ones(self.action_size) * self.epsilon / self.action_size
        policy[np.argmax(self.Q[next_state])] += 1 - self.epsilon
        expected_value = np.sum(policy * self.Q[next_state])
        
        # Calculate TD error
        td_error = reward + self.discount_factor * expected_value - self.Q[state, action]
        
        # Update action-value estimate
        self.Q[state, action] += self.learning_rate * td_error
```

## 4. Implementation Considerations

### 4.1 Experience Replay

```python
class ExperienceReplay:
    def __init__(self, capacity: int = 10000):
        """
        Initialize experience replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, experience: Tuple):
        """
        Add experience to buffer
        
        Args:
            experience: (state, action, reward, next_state) tuple
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch of experiences
        
        Args:
            batch_size: Size of batch
            
        Returns:
            List of experiences
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
```

### 4.2 Target Networks

```python
class TargetNetwork:
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize target network
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize main and target networks
        self.main_network = np.zeros((state_size, action_size))
        self.target_network = np.zeros((state_size, action_size))
        
        # Copy initial weights
        self.update_target()
    
    def update_target(self):
        """Update target network weights"""
        self.target_network = self.main_network.copy()
    
    def soft_update(self, tau: float = 0.001):
        """
        Soft update target network weights
        
        Args:
            tau: Update rate
        """
        self.target_network = (1 - tau) * self.target_network + tau * self.main_network
```

## 5. Practical Applications

### 5.1 Game Playing

TD methods are particularly effective in games with:
- Continuous state spaces
- Immediate rewards
- Need for online learning
- Real-time decision making

### 5.2 Robotics

In robotics applications, TD methods can be used for:
- Real-time control
- Continuous learning
- Adaptive behavior
- Multi-step planning

## 6. Advantages and Disadvantages

### 6.1 Advantages

1. **Online Learning**: Can learn from each step
2. **Model-Free**: No need for environment model
3. **Efficient Updates**: Updates estimates immediately
4. **Works with Continuing Tasks**: Can handle non-episodic problems

### 6.2 Disadvantages

1. **Bias in Estimates**: Due to bootstrapping
2. **Sensitivity to Parameters**: Learning rate and discount factor
3. **Initialization Dependent**: Performance depends on initial estimates
4. **Local Optima**: May get stuck in suboptimal policies

## 7. Best Practices

1. **Parameter Tuning**:
   - Learning rate scheduling
   - Discount factor selection
   - Exploration rate decay

2. **Value Function Approximation**:
   - Feature engineering
   - Neural network architecture
   - Regularization techniques

3. **Stability Improvements**:
   - Experience replay
   - Target networks
   - Gradient clipping

## 8. Future Directions

1. **Deep TD Learning**:
   - Deep Q-Networks (DQN)
   - Double DQN
   - Dueling DQN

2. **Distributional TD**:
   - C51
   - QR-DQN
   - IQN

3. **Multi-Agent TD**:
   - Independent Q-Learning
   - Centralized Training
   - Decentralized Execution

## 9. References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
2. Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
3. Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems. Technical Report CUED/F-INFENG/TR 166, Cambridge University.
4. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning. In AAAI (pp. 2094-2100). 