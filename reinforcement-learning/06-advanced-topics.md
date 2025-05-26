# Advanced Topics in Reinforcement Learning

## Introduction

This document covers advanced topics in reinforcement learning, including multi-agent systems, hierarchical reinforcement learning, inverse reinforcement learning, and other cutting-edge approaches.

## Multi-Agent Reinforcement Learning

### 1. Independent Q-Learning
```python
class IndependentQLearning:
    def __init__(
        self,
        num_agents: int,
        state_dims: List[int],
        action_dims: List[int],
        learning_rate: float,
        gamma: float,
        epsilon: float
    ):
        """
        Initialize Independent Q-Learning.

        Args:
            num_agents: Number of agents
            state_dims: State dimensions for each agent
            action_dims: Action dimensions for each agent
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.num_agents = num_agents
        self.q_networks = [
            QNetwork(state_dim, action_dim)
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        self.optimizers = [
            optim.Adam(net.parameters(), lr=learning_rate)
            for net in self.q_networks
        ]
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(
        self,
        states: List[torch.Tensor],
        training: bool = True
    ) -> List[torch.Tensor]:
        """
        Select actions for all agents.

        Args:
            states: List of states for each agent
            training: Whether in training mode

        Returns:
            List of selected actions
        """
        actions = []
        for i, (state, q_net) in enumerate(zip(states, self.q_networks)):
            if training and random.random() < self.epsilon:
                action = torch.randint(
                    0,
                    self.q_networks[i].action_dim,
                    (1,)
                )
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action = q_values.argmax()
            actions.append(action)
        return actions

    def update(
        self,
        experiences: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]]
    ) -> List[float]:
        """
        Update Q-networks for all agents.

        Args:
            experiences: List of (state, action, reward, next_state, done) tuples

        Returns:
            List of losses for each agent
        """
        losses = []
        for i, (state, action, reward, next_state, done) in enumerate(experiences):
            # Compute target Q-value
            with torch.no_grad():
                next_q_values = self.q_networks[i](next_state)
                target_q = reward + (1 - done) * self.gamma * next_q_values.max()
            
            # Compute current Q-value
            current_q = self.q_networks[i](state)[action]
            
            # Compute loss and update
            loss = F.mse_loss(current_q, target_q)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
            
            losses.append(loss.item())
        
        return losses
```

### 2. MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
```python
class MADDPG:
    def __init__(
        self,
        num_agents: int,
        state_dims: List[int],
        action_dims: List[int],
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float
    ):
        """
        Initialize MADDPG.

        Args:
            num_agents: Number of agents
            state_dims: State dimensions for each agent
            action_dims: Action dimensions for each agent
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Target network update rate
        """
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        
        # Initialize actors and critics
        self.actors = [
            Actor(state_dim, action_dim)
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        self.critics = [
            Critic(sum(state_dims), sum(action_dims))
            for _ in range(num_agents)
        ]
        
        # Initialize target networks
        self.target_actors = [
            Actor(state_dim, action_dim)
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        self.target_critics = [
            Critic(sum(state_dims), sum(action_dims))
            for _ in range(num_agents)
        ]
        
        # Initialize optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=actor_lr)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=critic_lr)
            for critic in self.critics
        ]
        
        # Initialize target networks
        self._update_targets()

    def select_action(
        self,
        states: List[torch.Tensor],
        noise: float = 0.0
    ) -> List[torch.Tensor]:
        """
        Select actions for all agents.

        Args:
            states: List of states for each agent
            noise: Action noise

        Returns:
            List of selected actions
        """
        actions = []
        for i, (state, actor) in enumerate(zip(states, self.actors)):
            with torch.no_grad():
                action = actor(state)
                if noise > 0:
                    action += torch.randn_like(action) * noise
                    action = torch.clamp(action, -1, 1)
            actions.append(action)
        return actions

    def update(
        self,
        experiences: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]]
    ) -> Tuple[List[float], List[float]]:
        """
        Update actors and critics for all agents.

        Args:
            experiences: List of (state, action, reward, next_state, done) tuples

        Returns:
            Tuple of (actor losses, critic losses)
        """
        actor_losses = []
        critic_losses = []
        
        # Update critics
        for i in range(self.num_agents):
            states = torch.cat([exp[0] for exp in experiences])
            actions = torch.cat([exp[1] for exp in experiences])
            rewards = torch.tensor([exp[2] for exp in experiences])
            next_states = torch.cat([exp[3] for exp in experiences])
            dones = torch.tensor([exp[4] for exp in experiences])
            
            # Compute target Q-value
            with torch.no_grad():
                next_actions = torch.cat([
                    self.target_actors[j](next_states)
                    for j in range(self.num_agents)
                ], dim=1)
                target_q = self.target_critics[i](
                    next_states,
                    next_actions
                )
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            # Compute current Q-value
            current_q = self.critics[i](states, actions)
            
            # Update critic
            critic_loss = F.mse_loss(current_q, target_q)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            critic_losses.append(critic_loss.item())
        
        # Update actors
        for i in range(self.num_agents):
            states = torch.cat([exp[0] for exp in experiences])
            actions = self.actors[i](states)
            
            # Compute actor loss
            actor_loss = -self.critics[i](
                states,
                torch.cat([
                    actions if j == i else exp[1]
                    for j, exp in enumerate(experiences)
                ], dim=1)
            ).mean()
            
            # Update actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            actor_losses.append(actor_loss.item())
        
        # Update target networks
        self._update_targets()
        
        return actor_losses, critic_losses

    def _update_targets(self):
        """Update target networks."""
        for i in range(self.num_agents):
            for target_param, param in zip(
                self.target_actors[i].parameters(),
                self.actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data +
                    (1 - self.tau) * target_param.data
                )
            
            for target_param, param in zip(
                self.target_critics[i].parameters(),
                self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data +
                    (1 - self.tau) * target_param.data
                )
```

## Hierarchical Reinforcement Learning

### 1. Options Framework
```python
class Option:
    def __init__(
        self,
        policy: nn.Module,
        termination: nn.Module,
        init_set: Set[State]
    ):
        """
        Initialize option.

        Args:
            policy: Option policy
            termination: Termination function
            init_set: Set of states where option can be initiated
        """
        self.policy = policy
        self.termination = termination
        self.init_set = init_set

class HierarchicalAgent:
    def __init__(
        self,
        options: List[Option],
        meta_policy: nn.Module,
        learning_rate: float,
        gamma: float
    ):
        """
        Initialize hierarchical agent.

        Args:
            options: List of options
            meta_policy: Meta-policy for selecting options
            learning_rate: Learning rate
            gamma: Discount factor
        """
        self.options = options
        self.meta_policy = meta_policy
        self.optimizer = optim.Adam(meta_policy.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_option(
        self,
        state: torch.Tensor,
        available_options: List[int]
    ) -> int:
        """
        Select option using meta-policy.

        Args:
            state: Current state
            available_options: List of available option indices

        Returns:
            Selected option index
        """
        with torch.no_grad():
            option_probs = self.meta_policy(state)
            option_probs = option_probs[available_options]
            option_idx = available_options[option_probs.argmax()]
        return option_idx

    def execute_option(
        self,
        option_idx: int,
        state: torch.Tensor,
        env: Environment
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Execute selected option.

        Args:
            option_idx: Option index
            state: Current state
            env: Environment

        Returns:
            Tuple of (next state, reward, done)
        """
        option = self.options[option_idx]
        done = False
        total_reward = 0
        
        while not done:
            # Select primitive action
            action = option.policy(state)
            next_state, reward, env_done = env.step(action)
            
            # Check option termination
            terminate = option.termination(next_state)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            done = env_done or terminate
        
        return state, total_reward, done

    def update(
        self,
        states: List[torch.Tensor],
        option_indices: List[int],
        rewards: List[float],
        next_states: List[torch.Tensor]
    ) -> float:
        """
        Update meta-policy.

        Args:
            states: List of states
            option_indices: List of selected option indices
            rewards: List of rewards
            next_states: List of next states

        Returns:
            Policy loss
        """
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Compute policy loss
        option_probs = self.meta_policy(torch.stack(states))
        selected_probs = option_probs[range(len(option_indices)), option_indices]
        policy_loss = -(selected_probs * returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()
```

## Inverse Reinforcement Learning

### 1. Maximum Entropy IRL
```python
class MaxEntIRL:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int,
        learning_rate: float,
        gamma: float
    ):
        """
        Initialize Maximum Entropy IRL.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            feature_dim: Feature dimension
            learning_rate: Learning rate
            gamma: Discount factor
        """
        self.reward_network = RewardNetwork(feature_dim)
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.reward_optimizer = optim.Adam(
            self.reward_network.parameters(),
            lr=learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )
        self.gamma = gamma

    def compute_feature_expectations(
        self,
        trajectories: List[List[Tuple[State, Action]]]
    ) -> torch.Tensor:
        """
        Compute feature expectations from expert trajectories.

        Args:
            trajectories: List of expert trajectories

        Returns:
            Feature expectations
        """
        feature_expectations = torch.zeros(self.reward_network.feature_dim)
        
        for trajectory in trajectories:
            for state, _ in trajectory:
                features = self._extract_features(state)
                feature_expectations += features
        
        return feature_expectations / len(trajectories)

    def compute_expected_features(
        self,
        policy: Policy,
        env: Environment,
        num_trajectories: int
    ) -> torch.Tensor:
        """
        Compute expected features under current policy.

        Args:
            policy: Current policy
            env: Environment
            num_trajectories: Number of trajectories to sample

        Returns:
            Expected features
        """
        feature_expectations = torch.zeros(self.reward_network.feature_dim)
        
        for _ in range(num_trajectories):
            state = env.reset()
            done = False
            
            while not done:
                action = policy.select_action(state)
                features = self._extract_features(state)
                feature_expectations += features
                
                state, _, done = env.step(action)
        
        return feature_expectations / num_trajectories

    def update(
        self,
        expert_trajectories: List[List[Tuple[State, Action]]],
        env: Environment,
        num_samples: int
    ) -> Tuple[float, float]:
        """
        Update reward and policy networks.

        Args:
            expert_trajectories: List of expert trajectories
            env: Environment
            num_samples: Number of policy samples

        Returns:
            Tuple of (reward loss, policy loss)
        """
        # Compute expert feature expectations
        expert_features = self.compute_feature_expectations(expert_trajectories)
        
        # Compute policy feature expectations
        policy_features = self.compute_expected_features(
            self.policy_network,
            env,
            num_samples
        )
        
        # Update reward network
        reward_loss = F.mse_loss(policy_features, expert_features)
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()
        
        # Update policy network
        policy_loss = -self.reward_network(policy_features).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return reward_loss.item(), policy_loss.item()

    def _extract_features(self, state: State) -> torch.Tensor:
        """
        Extract features from state.

        Args:
            state: Current state

        Returns:
            Feature vector
        """
        # Implement feature extraction
        pass
```

## Next Steps

1. Study [Implementation Guidelines](07-implementation.md)
2. Learn about [Applications](08-applications.md)
3. Explore [Research Directions](09-research.md) 