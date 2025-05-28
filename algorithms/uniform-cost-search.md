# Uniform Cost Search (UCS) in Intelligent Software Agents

## 1. Example: Finding the Cheapest Path in a Weighted Graph

Imagine an agent navigating a weighted graph, where each edge has a cost. The agent must find the path with the lowest total cost from a start node to a goal node. Uniform Cost Search (UCS) is ideal for this task because it expands the least-cost node first, ensuring the optimal path.

## 2. Applications of Uniform Cost Search

- **Pathfinding in Robotics and Games**: Finding the cheapest path in weighted environments.
- **Network Routing**: Optimizing routes in communication networks.
- **Resource Allocation**: Minimizing costs in resource distribution.
- **Scheduling**: Finding optimal schedules with minimal cost.
- **Game Playing**: Move selection in games with weighted actions.

## 3. Detailed Introduction

### 3.1 What is Uniform Cost Search?

UCS is a graph traversal algorithm that expands the least-cost node first. It uses a priority queue to manage the order of exploration, ensuring that nodes are processed in order of increasing cost.

### 3.2 How UCS Works

1. Start at the root node (or any node).
2. Use a priority queue to explore nodes in order of increasing cost.
3. Expand the least-cost node and update the costs of its neighbors.
4. Repeat until the goal is reached or all nodes are explored.

### 3.3 Python Example

```python
import heapq

def uniform_cost_search(graph, start, goal):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node == goal:
            return path + [node]
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [node]))
    return None
```

### 3.4 When to Use UCS

- When the goal is to find the cheapest path in a weighted graph.
- For problems where the cost of each action is known and positive.

### 3.5 Limitations

- Not suitable for graphs with negative weights (use Bellman-Ford instead).
- Can be memory-intensive for large graphs due to the priority queue.

### 3.6 Further Reading

- Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach. Pearson.
- Cormen, T. H., et al. (2009). Introduction to Algorithms. MIT Press. 