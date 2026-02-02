# graph + nodes + edges
from typing import List
from graph.facts import FactNode

class FactGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def build_edges_same_chunk(self):
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j and self.nodes[i].chunk_id == self.nodes[j].chunk_id:
                    self.nodes[i].neighbors.append(self.nodes[j])

    def propagate(self, alpha=0.7):
        for node in self.nodes:
            if not node.neighbors:
                continue
            neighbor_avg = sum(n.score for n in node.neighbors) / len(node.neighbors)
            node.score = alpha * node.score + (1 - alpha) * neighbor_avg

    def top_k(self, k=5):
        return sorted(self.nodes, key=lambda x: x.score, reverse=True)[:k]