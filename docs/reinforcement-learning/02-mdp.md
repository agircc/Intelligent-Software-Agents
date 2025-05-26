# Markov Decision Processes (MDP)

## Introduction

A Markov Decision Process (MDP) is the mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. It provides a formal way to model sequential decision-making problems.

## Core Components

### 1. States (S)
- Set of possible states in the environment
- Can be finite or infinite
- Must satisfy the Markov property

### 2. Actions (A)
- Set of possible actions
- Can be state-dependent
- Can be discrete or continuous

### 3. Transition Probabilities (P)
- P(s'|s,a): Probability of transitioning to state s' from state s after taking action a
- Must sum to 1 for each state-action pair
- Represents the environment dynamics

### 4. Reward Function (R)
- R(s,a,s'): Expected reward for taking action a in state s and transitioning to s'
- Can be deterministic or stochastic
- Defines the goal of the agent

### 5. Discount Factor (γ)
- γ ∈ [0,1]: Determines the importance of future rewards
- γ = 0: Only immediate rewards matter
- γ = 1: Future rewards are as important as immediate rewards

## Mathematical Formulation

### 1. MDP Tuple
```python
MDP = (S, A, P, R, γ)
```

### 2. State Transition
```python
P(s'|s,a) = Pr{S_t+1 = s' | S_t = s, A_t = a}
```

### 3. Reward Function
```python
R(s,a,s') = E[R_t+1 | S_t = s, A_t = a, S_t+1 = s']
```

### 4. Return
```python
G_t = R_t+1 + γR_t+2 + γ²R_t+3 + ... = Σ(γ^k * R_t+k+1)
```

## Implementation

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
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = discount_factor

    def get_transition_probability(
        self,
        state: State,
        action: Action,
        next_state: State
    ) -> float:
        """
        Get transition probability P(s'|s,a).

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Transition probability
        """
        return self.transitions.get((state, action), {}).get(next_state, 0.0)

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State
    ) -> float:
        """
        Get reward R(s,a,s').

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Expected reward
        """
        return self.rewards.get((state, action, next_state), 0.0)
```

## Properties

### 1. Markov Property
- Future states depend only on the current state and action
- Independent of past states and actions
- Memoryless property

### 2. Stationarity
- Transition probabilities and rewards don't change over time
- Environment dynamics remain constant
- Allows for consistent learning

### 3. Ergodic MDPs
- Every state is reachable from every other state
- Guarantees convergence of value iteration
- Ensures optimal policy exists

## Solving MDPs

### 1. Value Iteration
```python
def value_iteration(mdp: MDP, epsilon: float) -> Dict[State, float]:
    """
    Value iteration algorithm.

    Args:
        mdp: MDP instance
        epsilon: Convergence threshold

    Returns:
        Optimal value function
    """
    V = {s: 0.0 for s in mdp.states}
    
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            V[s] = max(
                sum(
                    mdp.get_transition_probability(s, a, s_prime) * (
                        mdp.get_reward(s, a, s_prime) +
                        mdp.gamma * V[s_prime]
                    )
                    for s_prime in mdp.states
                )
                for a in mdp.actions
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < epsilon:
            break
    
    return V
```

### 2. Policy Iteration
```python
def policy_iteration(mdp: MDP) -> Dict[State, Action]:
    """
    Policy iteration algorithm.

    Args:
        mdp: MDP instance

    Returns:
        Optimal policy
    """
    policy = {s: random.choice(list(mdp.actions)) for s in mdp.states}
    
    while True:
        # Policy evaluation
        V = evaluate_policy(mdp, policy)
        
        # Policy improvement
        policy_stable = True
        for s in mdp.states:
            old_action = policy[s]
            policy[s] = max(
                mdp.actions,
                key=lambda a: sum(
                    mdp.get_transition_probability(s, a, s_prime) * (
                        mdp.get_reward(s, a, s_prime) +
                        mdp.gamma * V[s_prime]
                    )
                    for s_prime in mdp.states
                )
            )
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy
```

## Extensions

### 1. Partially Observable MDPs (POMDPs)
- States are not fully observable
- Agent receives observations instead
- More complex but more realistic

### 2. Continuous MDPs
- Continuous state and action spaces
- Requires function approximation
- Common in robotics applications

### 3. Multi-Agent MDPs
- Multiple agents interacting
- Complex dynamics and rewards
- Game theory considerations

## Next Steps

1. Learn about [Value Functions](03-value-functions.md)
2. Study [Basic Algorithms](04-basic-algorithms.md)
3. Explore [Advanced Topics](05-advanced-topics.md) 