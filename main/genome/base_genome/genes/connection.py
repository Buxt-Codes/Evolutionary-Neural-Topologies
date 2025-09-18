import random

class ConnectionGene:
    """Represents a connection between two nodes."""
    def __init__(self, id, in_node, out_node, weight=None, enabled=True):
        self.id = id
        self.in_node = in_node
        self.out_node = out_node
        self.weight = random.uniform(-0.5, 0.5) if weight is None else weight
        self.enabled = enabled
        
    def __repr__(self):
        status = "E" if self.enabled else "D"
        return f"ConnectionGene({self.in_node}->{self.out_node}, w={self.weight:.2f}, {status})"