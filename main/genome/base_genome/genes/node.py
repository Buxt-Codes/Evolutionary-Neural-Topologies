class NodeGene:
    """Represents a node (neuron) in the network."""
    def __init__(self, id, bias=None, node_type="hidden", activation="relu"):
        self.id = id
        self.bias = None
        self.type = node_type  # "input", "hidden", "output"
        self.layer = -1
        self.activation = activation
        self.avg_activations = None

    def __repr__(self):
        return f"NodeGene(id={self.id}, layer={self.layer}, type='{self.type}')"