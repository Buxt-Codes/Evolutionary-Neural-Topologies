import copy
import torch
import torch.nn as nn
from collections import defaultdict, deque
from typing import List, Dict, Tuple

from .base_genome import BaseGenome, ConnectionGene, NodeGene
from ..config import Config

class GenotypeNet(nn.Module):
    def __init__(self, genotype, config):
        """Inisialise a GenotypeNet from a BaseGenome."""
        super().__init__()
        self.genotype = genotype
        self.use_bias = getattr(config, "use_bias", False)

        # Keep only enabled connections
        self.connections = [c for c in genotype.connections.values() if c.enabled]

        # Build node map
        self.nodes = {n.id: n for n in genotype.nodes.values()}
        self.sorted_nodes = sorted(genotype.nodes.values(), key=lambda n: (n.layer, n.id))
        self.node_index = {nid: idx for idx, nid in enumerate(self.nodes)}

        # Register weights
        self.weights = nn.Parameter(torch.tensor([c.weight for c in self.connections], dtype=torch.float32))

        # Register biases
        if self.use_bias:
            self.biases = nn.Parameter(torch.zeros(len(self.nodes)))
        else:
            self.register_buffer("biases", torch.zeros(len(self.nodes)))

        # Build edge index tensors
        src = [self.node_index[c.in_node] for c in self.connections]
        dst = [self.node_index[c.out_node] for c in self.connections]
        self.register_buffer("src_idx", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst_idx", torch.tensor(dst, dtype=torch.long))

        # Input/output nodes
        self.input_nodes = [n for n in genotype.nodes.values() if n.type == "input"]
        self.output_nodes = [n for n in genotype.nodes.values() if n.type == "output"]

        # Buffers for activation stats
        self.register_buffer("_activation_sums", torch.zeros(len(self.nodes)))
        self.register_buffer("_activation_counts", torch.zeros(len(self.nodes)))

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        values = torch.zeros(batch_size, len(self.nodes), device=device)

        # Fill input activations
        for i, n in enumerate(self.input_nodes):
            values[:, self.node_index[n.id]] = x[:, i]

        # Vectorized propagation
        for node in self.sorted_nodes:
            if node.type == "input":
                continue

            idx = self.node_index[node.id]

            # Gather contributions from all incoming edges in one go
            mask = (self.dst_idx == idx)
            if mask.any():
                contrib = values[:, self.src_idx[mask]] * self.weights[mask]
                total = contrib.sum(dim=1)
            else:
                total = torch.zeros(batch_size, device=device)

            # Bias
            if self.use_bias:
                total = total + self.biases[idx]

            # Activation
            values[:, idx] = torch.relu(total)

        # Record activations
        batch_means = values.mean(dim=0).detach()
        self._activation_sums += batch_means
        self._activation_counts += 1

        # Gather outputs
        out_idx = [self.node_index[n.id] for n in self.output_nodes]
        return values[:, out_idx]

    def export_genotype(self):
        """
        Return a new genotype with updated weights, biases, and avg activations.
        """
        new_genotype = copy.deepcopy(self.genotype)

        # Update connection weights
        enabled_connections = [c for c in new_genotype.connections.values() if c.enabled]
        for i, conn in enumerate(enabled_connections):
            conn.weight = self.weights[i].detach().item()

        # Update biases + activations
        for node in new_genotype.nodes.values():
            idx = self.node_index[node.id]

            if self.use_bias and node.type != "input":
                node.bias = self.biases[idx].detach().item()
            else:
                node.bias = None

            if self._activation_counts[idx] > 0:
                node.avg_activation = (self._activation_sums[idx] / self._activation_counts[idx]).item()
            else:
                node.avg_activation = 0.0

        return new_genotype