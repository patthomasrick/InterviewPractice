from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Deque, Dict, Hashable, List, Set


@dataclass
class Node:
    value: Hashable

    def __hash__(self) -> int:
        return hash(self.value)


class GraphVisitor(ABC):
    """
    Visitor for a graph traversal.
    """

    @abstractmethod
    def visitNodes(self, nodes: List[Node]):
        """Add nodes to the list of nodes to visit.

        Args:
            nodes (List[Node]): Nodes to visit.
        """
        pass

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Node:
        pass


class DepthFirstSearch(GraphVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._nodes: Deque[Node] = deque()
        self._visited: Set[Node] = set()

    def visitNodes(self, nodes: List[Node]):
        self._nodes.extend([n for n in nodes if n not in self._visited])
        self._visited.update(nodes)

    def __next__(self):
        if len(self._nodes) == 0:
            raise StopIteration
        return self._nodes.pop()


class BreadthFirstSearch(GraphVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._nodes: Deque[Node] = deque()
        self._visited: Set[Node] = set()

    def visitNodes(self, nodes: List[Node]):
        self._nodes.extend([n for n in nodes if n not in self._visited])
        self._visited.update(nodes)

    def __next__(self):
        if len(self._nodes) == 0:
            raise StopIteration
        return self._nodes.popleft()


class Graph(ABC):
    def traverse(self, visitor: GraphVisitor, start_node: Node) -> List[Any]:
        visitor.visitNodes([start_node])
        output = []
        for n in visitor:
            output.append(n.value)
            visitor.visitNodes(self.getNeighbors(n))
        return output

    @abstractmethod
    def addNode(self, node: Node):
        pass

    @abstractmethod
    def removeNode(self, node: Node) -> bool:
        pass

    @abstractmethod
    def addEdge(self, node_from: Node, node_to: Node) -> bool:
        pass

    @abstractmethod
    def removeEdge(self, node_from: Node, node_to: Node) -> bool:
        pass

    @abstractmethod
    def getNeighbors(self, node: Node) -> List[Node]:
        pass


@dataclass
class AdjacencyMatrixGraph(Graph):
    is_directed: bool
    num_nodes: int
    nodes: Dict[Node, int]
    matrix: List[List[bool]]

    def __init__(self, directed: bool = True, base_size: int = 0):
        self.is_directed = directed
        self.num_nodes = base_size
        self.nodes: Dict[Node, int] = dict()
        self.matrix: List[List[bool]] = []

    def _grow(self):
        self.matrix.append([False for _ in range(self.num_nodes)])
        for i in range(len(self.matrix)):
            self.matrix[i].append(False)
        self.num_nodes += 1

    def addNode(self, node: Node):
        self.nodes[node] = self.num_nodes
        self._grow()

    def removeNode(self, node: Node) -> bool:
        if node not in self.nodes:
            return False

        # Remove all edges in col/row.
        index = self.nodes[node]
        self.matrix[index] = [False for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            self.matrix[i][index] = False

        # Don't remove the column and row since that would be expensive. Just
        # forget about it for now.
        return True

    def addEdge(self, node_from: Node, node_to: Node) -> bool:
        return self._setEdge(node_from, node_to, True)

    def removeEdge(self, node_from: Node, node_to: Node) -> bool:
        return self._setEdge(node_from, node_to, False)

    def _setEdge(self, node_from: Node, node_to: Node, value: bool) -> bool:
        if node_from not in self.nodes or node_to not in self.nodes:
            return False

        from_index = self.nodes[node_from]
        to_index = self.nodes[node_to]

        # If we're directed, then sort the indices.
        if self.is_directed:
            from_index, to_index = sorted([from_index, to_index])

        self.matrix[from_index][to_index] = value

        return True

    def getNeighbors(self, node: Node) -> List[Node]:
        if node not in self.nodes:
            return []

        from_index = self.nodes[node]
        output = []
        for to_node, to_index in self.nodes.items():
            if self.matrix[from_index][to_index]:
                output.append(to_node)
        return output


class AdjacencyListGraph(Graph):
    pass


if __name__ == "__main__":
    g: Graph = AdjacencyMatrixGraph()

    # Make some nodes.
    n1 = Node("apple")
    n2 = Node("banana")
    n3 = Node("orange")
    n4 = Node("cars")

    # Add to graph.
    g.addNode(n1)
    g.addNode(n2)
    g.addNode(n3)
    g.addNode(n4)

    # Add some edges.
    g.addEdge(n1, n2)
    g.addEdge(n1, n3)
    g.addEdge(n2, n4)

    pprint(g)

    # Print a BFS traversal.
    bfs_visitor = BreadthFirstSearch()
    bfs_traversal = g.traverse(bfs_visitor, n1)
    pprint(bfs_traversal)

    # Print a DFS traversal.
    dfs_visitor = DepthFirstSearch()
    dfs_traversal = g.traverse(dfs_visitor, n1)
    pprint(dfs_traversal)
