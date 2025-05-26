# Value Functions

## Introduction

Value functions are fundamental components in reinforcement learning that estimate how good it is for an agent to be in a particular state or to take a particular action in a state. They are used to evaluate and improve policies.

## Types of Value Functions

### 1. State-Value Function V(s)
- Estimates the expected return from state s
- Represents the long-term value of being in a state
- Used to evaluate states

### 2. Action-Value Function Q(s,a)
- Estimates the expected return from taking action a in state s
- Represents the long-term value of taking an action
- Used to evaluate actions

## Mathematical Formulation

### 1. State-Value Function
```python
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ(γ^k * R_t+k+1) | S_t = s]
```

### 2. Action-Value Function
```python
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[Σ(γ^k * R_t+k+1) | S_t = s, A_t = a]
```

## Implementation

### 1. State-Value Function
```python
class StateValueFunction:
    def __init__(
        self,
        states: Set[State],
        discount_factor: float
    ):
        """
        Initialize state-value function.

        Args:
            states: Set of possible states
            discount_factor: Discount factor γ
        """
        self.values = {s: 0.0 for s in states}
        self.gamma = discount_factor

    def get_value(self, state: State) -> float:
        """
        Get value for a state.

        Args:
            state: Current state

        Returns:
            State value
        """
        return self.values.get(state, 0.0)

    def update(
        self,
        state: State,
        value: float
    ) -> None:
        """
        Update value for a state.

        Args:
            state: Current state
            value: New value
        """
        self.values[state] = value
```

### 2. Action-Value Function
```python
class ActionValueFunction:
    def __init__(
        self,
        states: Set[State],
        actions: Set[Action],
        discount_factor: float
    ):
        """
        Initialize action-value function.

        Args:
            states: Set of possible states
            actions: Set of possible actions
            discount_factor: Discount factor γ
        """
        self.values = {
            (s, a): 0.0
            for s in states
            for a in actions
        }
        self.gamma = discount_factor

    def get_value(
        self,
        state: State,
        action: Action
    ) -> float:
        """
        Get value for a state-action pair.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Action value
        """
        return self.values.get((state, action), 0.0)

    def update(
        self,
        state: State,
        action: Action,
        value: float
    ) -> None:
        """
        Update value for a state-action pair.

        Args:
            state: Current state
            action: Action taken
            value: New value
        """
        self.values[(state, action)] = value
```

## Value Function Approximation

### 1. Linear Function Approximation
```python
class LinearValueFunction:
    def __init__(
        self,
        feature_dim: int,
        learning_rate: float
    ):
        """
        Initialize linear value function.

        Args:
            feature_dim: Feature dimension
            learning_rate: Learning rate
        """
        self.weights = np.zeros(feature_dim)
        self.alpha = learning_rate

    def get_value(self, features: np.ndarray) -> float:
        """
        Get value for a feature vector.

        Args:
            features: Feature vector

        Returns:
            Approximated value
        """
        return np.dot(self.weights, features)

    def update(
        self,
        features: np.ndarray,
        target: float
    ) -> None:
        """
        Update weights using gradient descent.

        Args:
            features: Feature vector
            target: Target value
        """
        prediction = self.get_value(features)
        error = target - prediction
        self.weights += self.alpha * error * features
```

### 2. Neural Network Approximation
```python
class NeuralValueFunction:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        learning_rate: float
    ):
        """
        Initialize neural network value function.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
        """
        self.model = self._build_network(
            input_dim,
            hidden_dims,
            learning_rate
        )

    def _build_network(
        self,
        input_dim: int,
        hidden_dims: List[int],
        learning_rate: float
    ) -> tf.keras.Model:
        """Build neural network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation='relu')
            for dim in hidden_dims
        ] + [
            tf.keras.layers.Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mse'
        )
        return model

    def get_value(self, features: np.ndarray) -> float:
        """
        Get value for a feature vector.

        Args:
            features: Feature vector

        Returns:
            Approximated value
        """
        return self.model.predict(features.reshape(1, -1))[0][0]

    def update(
        self,
        features: np.ndarray,
        target: float
    ) -> None:
        """
        Update network using gradient descent.

        Args:
            features: Feature vector
            target: Target value
        """
        self.model.fit(
            features.reshape(1, -1),
            np.array([target]).reshape(1, 1),
            verbose=0
        )
```

## Value Function Updates

### 1. Monte Carlo Update
```python
def monte_carlo_update(
    value_function: Union[StateValueFunction, ActionValueFunction],
    episode: List[Tuple[State, Action, float]]
) -> None:
    """
    Update value function using Monte Carlo method.

    Args:
        value_function: Value function to update
        episode: List of (state, action, reward) tuples
    """
    returns = 0
    for state, action, reward in reversed(episode):
        returns = reward + value_function.gamma * returns
        if isinstance(value_function, StateValueFunction):
            value_function.update(state, returns)
        else:
            value_function.update(state, action, returns)
```

### 2. Temporal Difference Update
```python
def temporal_difference_update(
    value_function: Union[StateValueFunction, ActionValueFunction],
    state: State,
    action: Optional[Action],
    reward: float,
    next_state: State,
    next_action: Optional[Action],
    alpha: float
) -> None:
    """
    Update value function using TD method.

    Args:
        value_function: Value function to update
        state: Current state
        action: Current action (optional)
        reward: Current reward
        next_state: Next state
        next_action: Next action (optional)
        alpha: Learning rate
    """
    if isinstance(value_function, StateValueFunction):
        current_value = value_function.get_value(state)
        next_value = value_function.get_value(next_state)
        target = reward + value_function.gamma * next_value
        value_function.update(
            state,
            current_value + alpha * (target - current_value)
        )
    else:
        current_value = value_function.get_value(state, action)
        next_value = value_function.get_value(next_state, next_action)
        target = reward + value_function.gamma * next_value
        value_function.update(
            state,
            action,
            current_value + alpha * (target - current_value)
        )
```

## Next Steps

1. Study [Basic Algorithms](04-basic-algorithms.md)
2. Learn about [Policy Methods](05-policy-methods.md)
3. Explore [Advanced Topics](06-advanced-topics.md) 