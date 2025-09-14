import random

class ConnectionGene:
    """Represents a connection between two nodes."""
    def __init__(self, in_node, out_node, innovation, weight=None, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.innovation = innovation
        self.weight = random.uniform(-1.0, 1.0) if weight is None else weight
        self.enabled = enabled
        
    def __repr__(self):
        status = "E" if self.enabled else "D"
        return f"ConnectionGene(innov={self.innovation}, {self.in_node}->{self.out_node}, w={self.weight:.2f}, {status})"