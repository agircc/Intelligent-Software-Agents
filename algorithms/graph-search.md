# Graph Search Algorithms in Intelligent Software Agents

## 1. Example: Pathfinding in a Maze

Imagine an agent navigating a maze from a start point to a goal. The agent must find the shortest path, avoiding obstacles. Graph search algorithms like Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's Algorithm, and A* Search are commonly used to solve such problems efficiently.

## 2. Applications of Graph Search

- **Pathfinding in Robotics and Games**: Navigation for robots and game characters.
- **Network Routing**: Finding optimal paths in communication and transportation networks.
- **Planning and Scheduling**: Task ordering and resource allocation.
- **Web Crawling**: Systematic exploration of web pages.
- **Puzzle Solving**: Solving puzzles like the 8-puzzle or Rubik's Cube.

## 3. Detailed Introduction

### 3.1 What is Graph Search?

Graph search algorithms systematically explore nodes and edges in a graph to find paths, connected components, or optimal solutions. They are fundamental in AI for problem-solving and planning.

### 3.2 Core Algorithms

#### Breadth-First Search (BFS)
- Explores all neighbors at the current depth before moving to the next level.
- Guarantees shortest path in unweighted graphs.

**Python Example:**
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node] - visited)
    return visited
```

#### Depth-First Search (DFS)
- Explores as far as possible along each branch before backtracking.
- Useful for topological sorting and cycle detection.

**Python Example:**
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start] - visited:
        dfs(graph, neighbor, visited)
    return visited
```

#### Dijkstra's Algorithm
- Finds shortest paths from a source to all nodes in a weighted graph with non-negative weights.

**Python Example:**
```python
import heapq

def dijkstra(graph, start):
    queue = [(0, start)]
    distances = {start: 0}
    while queue:
        (cost, node) = heapq.heappop(queue)
        for neighbor, weight in graph[node]:
            new_cost = cost + weight
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
    return distances
```

#### A* Search
- Uses heuristics to guide the search, improving efficiency for many problems.

**Python Example:**
```python
import heapq

def astar(graph, start, goal, heuristic):
    queue = [(0 + heuristic(start), 0, start, [])]
    visited = set()
    while queue:
        (est_total, cost, node, path) = heapq.heappop(queue)
        if node == goal:
            return path + [node]
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (
                    cost + weight + heuristic(neighbor),
                    cost + weight,
                    neighbor,
                    path + [node]
                ))
    return None
```

### 3.3 When to Use Graph Search
- When the problem can be modeled as a graph (states, transitions).
- For finding paths, connectivity, or optimal solutions in discrete spaces.

### 3.4 Limitations
- Scalability issues for very large graphs.
- Heuristic design is critical for A* efficiency.

### 3.5 Further Reading
- Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach. Pearson.
- Cormen, T. H., et al. (2009). Introduction to Algorithms. MIT Press. 