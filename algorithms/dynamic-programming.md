# Dynamic Programming in Reinforcement Learning

## 1. Example: Gridworld Navigation

Imagine an agent navigating a 4x4 gridworld. The agent starts in the top-left corner and aims to reach the bottom-right corner, receiving a reward of +1 for reaching the goal and 0 otherwise. The environment is deterministic: each action (up, down, left, right) moves the agent in the intended direction unless it hits a wall. The agent wants to find the shortest path to the goal.

Dynamic Programming (DP) methods, such as Value Iteration and Policy Iteration, can be used to compute the optimal policy for this environment by systematically evaluating and improving value functions for all states.

## 2. Applications of Dynamic Programming

Dynamic Programming is foundational in reinforcement learning and is widely used in:

- **Optimal Control**: Solving Markov Decision Processes (MDPs) where the model is known.
- **Robotics**: Planning optimal paths and actions in known environments.
- **Operations Research**: Inventory management, resource allocation, and scheduling problems.
- **Game AI**: Computing optimal strategies in board games and puzzles.
- **Finance**: Portfolio optimization and option pricing when the system dynamics are known.

DP is best suited for problems where the environment's dynamics (transition probabilities and rewards) are fully known and the state/action spaces are not prohibitively large.

## 3. Detailed Introduction

### 3.1 What is Dynamic Programming?

Dynamic Programming is a collection of algorithms that solve complex problems by breaking them down into simpler subproblems and solving each subproblem only once, storing the solutions for reuse. In reinforcement learning, DP is used to compute value functions and optimal policies for Markov Decision Processes (MDPs) when the model is known.

### 3.2 Core DP Algorithms in RL

#### Value Iteration

Value Iteration is an iterative algorithm that updates the value of each state based on the Bellman optimality equation:

\[
V_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]
\]

where:
- \(V_k(s)\): Value of state \(s\) at iteration \(k\)
- \(p(s', r | s, a)\): Probability of transitioning to state \(s'\) and receiving reward \(r\) after taking action \(a\) in state \(s\)
- \(\gamma\): Discount factor

**Python Example:**

```python
import numpy as np

def value_iteration(P, R, gamma=0.9, theta=1e-6):
    """
    P: Transition probability matrix (S x A x S)
    R: Reward matrix (S x A x S)
    gamma: Discount factor
    theta: Convergence threshold
    """
    S, A, _ = P.shape
    V = np.zeros(S)
    while True:
        delta = 0
        for s in range(S):
            v = V[s]
            V[s] = max(
                sum(P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(S))
                for a in range(A)
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V
```

#### Policy Iteration

Policy Iteration alternates between policy evaluation (computing the value function for a fixed policy) and policy improvement (updating the policy greedily with respect to the value function):

1. **Policy Evaluation:**
   - Iteratively update \(V(s)\) for all states under the current policy.
2. **Policy Improvement:**
   - For each state, choose the action that maximizes the expected value.

**Python Example:**

```python
def policy_iteration(P, R, gamma=0.9, theta=1e-6):
    S, A, _ = P.shape
    policy = np.zeros(S, dtype=int)
    V = np.zeros(S)
    is_policy_stable = False
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(S):
                v = V[s]
                a = policy[s]
                V[s] = sum(P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(S))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        # Policy Improvement
        is_policy_stable = True
        for s in range(S):
            old_action = policy[s]
            policy[s] = np.argmax([
                sum(P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(S))
                for a in range(A)
            ])
            if old_action != policy[s]:
                is_policy_stable = False
    return policy, V
```

### 3.3 When to Use Dynamic Programming

- The environment's model (transition and reward functions) is fully known.
- The state and action spaces are small to moderate (DP scales poorly with large spaces).
- You need to compute the optimal policy or value function exactly (as a baseline or for planning).

### 3.4 Limitations

- **Scalability**: DP methods are not practical for large or continuous state/action spaces due to the "curse of dimensionality."
- **Model Requirement**: Requires full knowledge of the environment's dynamics.

### 3.5 Further Reading

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
- Bertsekas, D. P. (2012). Dynamic Programming and Optimal Control. Athena Scientific. 