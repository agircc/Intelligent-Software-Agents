# Basic Reinforcement Learning Algorithms

## Introduction

This document covers fundamental reinforcement learning algorithms that form the basis for more advanced methods. These algorithms are essential for understanding how agents learn to make decisions through interaction with their environment.

## Dynamic Programming Methods

### 1. Value Iteration
```python
def value_iteration(
    mdp: MDP,
    epsilon: float = 1e-6,
    max_iterations: int = 1000
) -> Dict[State, float]:
    """
    Value iteration algorithm for solving MDPs.

    Args:
        mdp: Markov Decision Process
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Optimal value function
    """
    V = {s: 0.0 for s in mdp.states}
    
    for _ in range(max_iterations):
        delta = 0
        for s in mdp.states:
            v = V[s]
            V[s] = max(
                sum(
                    mdp.get_transition_prob(s, a, s_prime) * 
                    (mdp.get_reward(s, a, s_prime) + mdp.gamma * V[s_prime])
                    for s_prime in mdp.states
                )
                for a in mdp.get_actions(s)
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < epsilon:
            break
    
    return V
```

### 2. Policy Iteration
```python
def policy_iteration(
    mdp: MDP,
    epsilon: float = 1e-6,
    max_iterations: int = 1000
) -> Tuple[Dict[State, Action], Dict[State, float]]:
    """
    Policy iteration algorithm for solving MDPs.

    Args:
        mdp: Markov Decision Process
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Optimal policy and value function
    """
    # Initialize random policy
    policy = {
        s: random.choice(list(mdp.get_actions(s)))
        for s in mdp.states
    }
    V = {s: 0.0 for s in mdp.states}
    
    for _ in range(max_iterations):
        # Policy evaluation
        while True:
            delta = 0
            for s in mdp.states:
                v = V[s]
                V[s] = sum(
                    mdp.get_transition_prob(s, policy[s], s_prime) * 
                    (mdp.get_reward(s, policy[s], s_prime) + mdp.gamma * V[s_prime])
                    for s_prime in mdp.states
                )
                delta = max(delta, abs(v - V[s]))
            
            if delta < epsilon:
                break
        
        # Policy improvement
        policy_stable = True
        for s in mdp.states:
            old_action = policy[s]
            policy[s] = max(
                mdp.get_actions(s),
                key=lambda a: sum(
                    mdp.get_transition_prob(s, a, s_prime) * 
                    (mdp.get_reward(s, a, s_prime) + mdp.gamma * V[s_prime])
                    for s_prime in mdp.states
                )
            )
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, V
```

## Monte Carlo Methods

### 1. First-Visit MC Prediction
```python
def first_visit_mc_prediction(
    env: Environment,
    policy: Policy,
    num_episodes: int,
    gamma: float
) -> Dict[State, float]:
    """
    First-visit Monte Carlo prediction for estimating value function.

    Args:
        env: Environment
        policy: Policy to evaluate
        num_episodes: Number of episodes
        gamma: Discount factor

    Returns:
        Estimated value function
    """
    V = defaultdict(float)
    returns = defaultdict(list)
    
    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        done = False
        
        while not done:
            action = policy.select_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        G = 0
        visited_states = set()
        
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G = gamma * G + reward
            
            if state not in visited_states:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited_states.add(state)
    
    return V
```

### 2. Monte Carlo Control
```python
def monte_carlo_control(
    env: Environment,
    num_episodes: int,
    gamma: float,
    epsilon: float
) -> Tuple[Dict[State, Action], Dict[Tuple[State, Action], float]]:
    """
    Monte Carlo control with epsilon-greedy exploration.

    Args:
        env: Environment
        num_episodes: Number of episodes
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Optimal policy and action-value function
    """
    Q = defaultdict(float)
    returns = defaultdict(list)
    policy = defaultdict(lambda: defaultdict(float))
    
    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(env.get_actions(state))
            else:
                action = max(
                    env.get_actions(state),
                    key=lambda a: Q[(state, a)]
                )
            
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        G = 0
        visited_sa = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if (state, action) not in visited_sa:
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
                
                # Update policy
                best_action = max(
                    env.get_actions(state),
                    key=lambda a: Q[(state, a)]
                )
                for a in env.get_actions(state):
                    policy[state][a] = epsilon / len(env.get_actions(state))
                policy[state][best_action] += 1 - epsilon
                
                visited_sa.add((state, action))
    
    return policy, Q
```

## Temporal Difference Methods

### 1. SARSA (State-Action-Reward-State-Action)
```python
def sarsa(
    env: Environment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float
) -> Tuple[Dict[State, Action], Dict[Tuple[State, Action], float]]:
    """
    SARSA algorithm for on-policy control.

    Args:
        env: Environment
        num_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Optimal policy and action-value function
    """
    Q = defaultdict(float)
    policy = defaultdict(lambda: defaultdict(float))
    
    for _ in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_action(
            state,
            env.get_actions(state),
            Q,
            epsilon
        )
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_action(
                next_state,
                env.get_actions(next_state),
                Q,
                epsilon
            )
            
            # Update Q-value
            Q[(state, action)] += alpha * (
                reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )
            
            # Update policy
            best_action = max(
                env.get_actions(state),
                key=lambda a: Q[(state, a)]
            )
            for a in env.get_actions(state):
                policy[state][a] = epsilon / len(env.get_actions(state))
            policy[state][best_action] += 1 - epsilon
            
            state = next_state
            action = next_action
    
    return policy, Q
```

### 2. Q-Learning
```python
def q_learning(
    env: Environment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float
) -> Tuple[Dict[State, Action], Dict[Tuple[State, Action], float]]:
    """
    Q-learning algorithm for off-policy control.

    Args:
        env: Environment
        num_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Optimal policy and action-value function
    """
    Q = defaultdict(float)
    policy = defaultdict(lambda: defaultdict(float))
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = epsilon_greedy_action(
                state,
                env.get_actions(state),
                Q,
                epsilon
            )
            next_state, reward, done = env.step(action)
            
            # Update Q-value
            next_value = max(
                Q[(next_state, a)]
                for a in env.get_actions(next_state)
            )
            Q[(state, action)] += alpha * (
                reward + gamma * next_value - Q[(state, action)]
            )
            
            # Update policy
            best_action = max(
                env.get_actions(state),
                key=lambda a: Q[(state, a)]
            )
            for a in env.get_actions(state):
                policy[state][a] = epsilon / len(env.get_actions(state))
            policy[state][best_action] += 1 - epsilon
            
            state = next_state
    
    return policy, Q
```

## Helper Functions

### 1. Epsilon-Greedy Action Selection
```python
def epsilon_greedy_action(
    state: State,
    actions: List[Action],
    Q: Dict[Tuple[State, Action], float],
    epsilon: float
) -> Action:
    """
    Select action using epsilon-greedy policy.

    Args:
        state: Current state
        actions: Available actions
        Q: Action-value function
        epsilon: Exploration rate

    Returns:
        Selected action
    """
    if random.random() < epsilon:
        return random.choice(actions)
    return max(
        actions,
        key=lambda a: Q[(state, a)]
    )
```

## Next Steps

1. Learn about [Policy Methods](05-policy-methods.md)
2. Explore [Advanced Topics](06-advanced-topics.md)
3. Study [Implementation Guidelines](07-implementation.md) 