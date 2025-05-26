# Policy Methods

## Introduction

Policy methods in reinforcement learning directly optimize the policy function that maps states to actions. These methods are particularly useful for continuous action spaces and can provide better convergence properties than value-based methods in certain scenarios.

## Policy Gradient Methods

### 1. REINFORCE Algorithm
```python
class REINFORCE:
    def __init__(
        self,
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float
    ):
        """
        Initialize REINFORCE algorithm.

        Args:
            policy_network: Neural network for policy
            optimizer: Optimizer for policy network
            gamma: Discount factor
        """
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            Selected action and its log probability
        """
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob

    def update(
        self,
        rewards: List[float],
        log_probs: List[torch.Tensor]
    ) -> float:
        """
        Update policy using REINFORCE algorithm.

        Args:
            rewards: List of rewards
            log_probs: List of action log probabilities

        Returns:
            Policy loss
        """
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()
```

### 2. Actor-Critic Methods
```python
class ActorCritic:
    def __init__(
        self,
        actor_network: nn.Module,
        critic_network: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float
    ):
        """
        Initialize Actor-Critic algorithm.

        Args:
            actor_network: Neural network for policy
            critic_network: Neural network for value function
            actor_optimizer: Optimizer for actor network
            critic_optimizer: Optimizer for critic network
            gamma: Discount factor
        """
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma

    def select_action(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            Selected action, its log probability, and state value
        """
        action_probs = self.actor_network(state)
        state_value = self.critic_network(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value

    def update(
        self,
        rewards: List[float],
        log_probs: List[torch.Tensor],
        state_values: List[torch.Tensor],
        next_value: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update actor and critic networks.

        Args:
            rewards: List of rewards
            log_probs: List of action log probabilities
            state_values: List of state values
            next_value: Value of next state

        Returns:
            Actor loss and critic loss
        """
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = next_value
        
        for r, v in zip(reversed(rewards), reversed(state_values)):
            R = r + self.gamma * R
            advantage = R - v
            returns.insert(0, R)
            advantages.insert(0, advantage)
        
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        actor_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            actor_loss.append(-log_prob * advantage)
        actor_loss = torch.stack(actor_loss).sum()
        
        critic_loss = []
        for value, R in zip(state_values, returns):
            critic_loss.append(F.mse_loss(value, R))
        critic_loss = torch.stack(critic_loss).sum()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
```

## Proximal Policy Optimization (PPO)

### 1. PPO Implementation
```python
class PPO:
    def __init__(
        self,
        actor_network: nn.Module,
        critic_network: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float
    ):
        """
        Initialize PPO algorithm.

        Args:
            actor_network: Neural network for policy
            critic_network: Neural network for value function
            actor_optimizer: Optimizer for actor network
            critic_optimizer: Optimizer for critic network
            gamma: Discount factor
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy loss coefficient
        """
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            Selected action, its log probability, and state value
        """
        action_probs = self.actor_network(state)
        state_value = self.critic_network(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: List[float],
        next_value: torch.Tensor,
        num_epochs: int
    ) -> Tuple[float, float]:
        """
        Update actor and critic networks using PPO.

        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old log probabilities
            rewards: List of rewards
            next_value: Value of next state
            num_epochs: Number of update epochs

        Returns:
            Actor loss and critic loss
        """
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = next_value
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        state_values = self.critic_network(states).squeeze()
        advantages = returns - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(num_epochs):
            # Calculate new action probabilities
            action_probs = self.actor_network(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratios and losses
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_pred = self.critic_network(states).squeeze()
            value_loss = F.mse_loss(value_pred, returns)
            
            # Total loss
            total_loss = (
                actor_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        return actor_loss.item(), value_loss.item()
```

## Trust Region Policy Optimization (TRPO)

### 1. TRPO Implementation
```python
class TRPO:
    def __init__(
        self,
        actor_network: nn.Module,
        critic_network: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float,
        max_kl: float,
        damping: float,
        value_coef: float,
        entropy_coef: float
    ):
        """
        Initialize TRPO algorithm.

        Args:
            actor_network: Neural network for policy
            critic_network: Neural network for value function
            critic_optimizer: Optimizer for critic network
            gamma: Discount factor
            max_kl: Maximum KL divergence
            damping: Damping coefficient
            value_coef: Value loss coefficient
            entropy_coef: Entropy loss coefficient
        """
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.max_kl = max_kl
        self.damping = damping
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            Selected action, its log probability, and state value
        """
        action_probs = self.actor_network(state)
        state_value = self.critic_network(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: List[float],
        next_value: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update actor and critic networks using TRPO.

        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old log probabilities
            rewards: List of rewards
            next_value: Value of next state

        Returns:
            Actor loss and critic loss
        """
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = next_value
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        state_values = self.critic_network(states).squeeze()
        advantages = returns - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate new action probabilities
        action_probs = self.actor_network(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr_loss = -(ratio * advantages).mean()
        
        # Calculate KL divergence
        kl = (old_log_probs - new_log_probs).mean()
        
        # Calculate value loss
        value_pred = self.critic_network(states).squeeze()
        value_loss = F.mse_loss(value_pred, returns)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        # TRPO update for actor
        grads = torch.autograd.grad(surr_loss, self.actor_network.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads])
        
        def Fvp(v):
            kl = (old_log_probs - new_log_probs).mean()
            kl_grads = torch.autograd.grad(kl, self.actor_network.parameters(), create_graph=True)
            kl_grad = torch.cat([grad.view(-1) for grad in kl_grads])
            Fvp = torch.autograd.grad(kl_grad.dot(v), self.actor_network.parameters())
            Fvp = torch.cat([grad.contiguous().view(-1) for grad in Fvp])
            return Fvp + self.damping * v
        
        # Conjugate gradient algorithm
        def conjugate_gradient(Fvp, b, nsteps=10):
            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            
            for i in range(nsteps):
                Ap = Fvp(p)
                alpha = r.dot(r) / (p.dot(Ap) + 1e-8)
                x += alpha * p
                r_new = r - alpha * Ap
                beta = r_new.dot(r_new) / (r.dot(r) + 1e-8)
                r = r_new
                p = r + beta * p
            
            return x
        
        # Calculate step direction
        step_dir = conjugate_gradient(Fvp, -loss_grad)
        
        # Calculate step size
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum()
        lm = torch.sqrt(shs / self.max_kl)
        full_step = step_dir / lm
        
        # Update actor parameters
        old_params = torch.cat([p.view(-1) for p in self.actor_network.parameters()])
        new_params = old_params + full_step
        
        # Update network
        start_idx = 0
        for param in self.actor_network.parameters():
            size = param.numel()
            param.data.copy_(new_params[start_idx:start_idx + size].view(param.size()))
            start_idx += size
        
        return surr_loss.item(), value_loss.item()
```

## Next Steps

1. Explore [Advanced Topics](06-advanced-topics.md)
2. Study [Implementation Guidelines](07-implementation.md)
3. Learn about [Applications](08-applications.md) 