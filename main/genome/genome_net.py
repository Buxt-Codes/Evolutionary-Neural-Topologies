import torch
import torch.nn as nn

from .base_genome import BaseGenome

class GenomeNet(nn.Module):
    """A PyTorch Module that represents the phenotype of a Genome."""
    def __init__(self, genome: Genome):
        super().__init__()
        self.genome = genome
        
        # Identify node types
        self.input_nodes = sorted([n.id for n in genome.nodes.values() if n.type == "input"], reverse=True)
        self.output_nodes = sorted([n.id for n in genome.nodes.values() if n.type == "output"])
        
        # Build network structure for forward pass
        self._build_network()

    def _get_activation(self, name: str):
        if name == 'tanh':
            return torch.tanh
        elif name == 'sigmoid':
            return torch.sigmoid
        elif name == 'relu':
            return torch.relu
        # Default to identity
        return lambda x: x

    def _build_network(self):
        """Topologically sort the nodes to create an execution plan."""
        self.node_eval_order = []
        
        # Create adjacency list and in-degree count for Kahn's algorithm
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        
        for conn in self.genome.connections.values():
            if conn.enabled:
                in_degree[conn.out_node] += 1
                adj[conn.in_node].append(conn.out_node)

        # Start with all nodes that have an in-degree of 0 (inputs)
        queue = [n_id for n_id in self.genome.nodes if in_degree[n_id] == 0]
        
        while queue:
            node_id = queue.pop(0)
            self.node_eval_order.append(node_id)
            
            for neighbor_id in sorted(adj[node_id]): # Sort for deterministic order
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        # Store connections and activations for the forward pass
        self.connections_map = defaultdict(list)
        self.activations = {}
        for conn in self.genome.connections.values():
             if conn.enabled:
                self.connections_map[conn.out_node].append((conn.in_node, torch.tensor(conn.weight, dtype=torch.float32)))
        
        for node in self.genome.nodes.values():
            self.activations[node.id] = self._get_activation(node.activation)

    def forward(self, x: torch.Tensor):
        """Executes the network in topologically sorted order."""
        if x.dim() == 1:
            x = x.unsqueeze(0) # Ensure batch dimension
        
        if x.shape[1] != len(self.input_nodes):
            raise ValueError(f"Input tensor size ({x.shape[1]}) does not match number of input nodes ({len(self.input_nodes)})")
        
        node_values = {}
        
        # Initialize input node values
        for i, node_id in enumerate(self.input_nodes):
            node_values[node_id] = x[:, i]

        # Evaluate nodes in topological order
        for node_id in self.node_eval_order:
            if node_id in self.input_nodes:
                continue
            
            # Sum inputs from incoming connections
            incoming_sum = torch.zeros(x.shape[0], dtype=torch.float32)
            if node_id in self.connections_map:
                for in_node_id, weight in self.connections_map[node_id]:
                    # Ensure the source node value has been computed
                    if in_node_id in node_values:
                        incoming_sum += node_values[in_node_id] * weight

            # Apply activation function
            node_values[node_id] = self.activations[node_id](incoming_sum)

        # Collect output values
        output_vals = [node_values.get(out_id, torch.zeros(x.shape[0])) for out_id in self.output_nodes]
        return torch.stack(output_vals, dim=1)