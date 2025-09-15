import random
import copy
from typing import Dict, List

from .genes import ConnectionGene, NodeGene
from ...config import Config

class BaseGenome:
    """Represents an individual's genetic makeup, defining a neural network."""
    def __init__(self, id: int, input_size: int, output_size: int, config: Config):
        self.id = id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness = 0.0
        self.node_idx = 0
        self.connection_idx = 0
        self.config = config

        # Create input and output nodes
        for _ in range(input_size):
            self.nodes[self.node_idx] = NodeGene(self.node_idx, node_type="input")
            self.node_idx += 1
        
        for _ in range(output_size):
            self.nodes[self.node_idx] = NodeGene(self.node_idx, node_type="output")
            self.node_idx += 1
        
        # Create hidden nodes
        num_hidden = random.randint(0, config.initial_max_hidden)
        for _ in range(num_hidden):
            self.nodes[self.node_idx] = NodeGene(self.node_idx, node_type="hidden")
            self.node_idx += 1

        self.input_nodes = [n for n in self.nodes if self.nodes[n].type == "input"]
        self.output_nodes = [n for n in self.nodes if self.nodes[n].type == "output"]
        hidden_nodes = [n for n in self.nodes if self.nodes[n].type == "hidden"]
        
        # Random initial connections
        if hidden_nodes:
            max_layers = config.initial_max_layers if len(hidden_nodes) > config.initial_max_layers else len(hidden_nodes)
            num_layers = random.randint(2, max_layers)
        else:
            num_layers = 2
        self.layers: Dict[int, List[int]] = {i: [] for i in range(num_layers + 1)}

        # Assign inputs to first layer
        for n in self.input_nodes:
            self.layers[0].append(n)
            self.nodes[n].layer = 0

        # Assign outputs to final layer
        for n in self.output_nodes:
            self.layers[num_layers].append(n)
            self.nodes[n].layer = num_layers

        # Assign hidden nodes randomly across intermediate layers
        for n in hidden_nodes:
            layer = random.randint(1, num_layers - 1)
            self.layers[layer].append(n)
            self.nodes[n].layer = layer

        for in_layer in range(num_layers):
            for in_node in self.layers[in_layer]:
                for out_layer in range(in_layer + 1, num_layers + 1):
                    for out_node in self.layers[out_layer]:
                        if random.random() < config.initial_connection_prob:
                            self.connections[self.connection_idx] = ConnectionGene(
                                id=self.connection_idx,
                                in_node=in_node,
                                out_node=out_node,
                                enabled=True
                            )
                            self.connection_idx += 1

        for node_id, node in self.nodes.items():
            # Ensure incoming for non-input nodes
            if node.type != "input":
                if not any(c.out_node == node_id for c in self.connections.values()):
                    # pick a random earlier layer
                    candidate_layers = [l for l in self.layers if l < self.nodes[node_id].layer]
                    if candidate_layers:
                        in_layer = random.choice(candidate_layers)
                        in_node = random.choice(self.layers[in_layer])
                        self.connections[self.connection_idx] = ConnectionGene(
                            id=self.connection_idx,
                            in_node=in_node,
                            out_node=node_id,
                            enabled=True
                        )
                        self.connection_idx += 1
            
            if node.type != "output":
                if not any(c.in_node == node_id for c in self.connections.values()):
                    # pick a random later layer
                    candidate_layers = [l for l in self.layers if l > self.nodes[node_id].layer]
                    if candidate_layers:
                        out_layer = random.choice(candidate_layers)
                        out_node = random.choice(self.layers[out_layer])
                        self.connections[self.connection_idx] = ConnectionGene(
                            id=self.connection_idx,
                            in_node=node_id,
                            out_node=out_node,
                            enabled=True
                        )
                        self.connection_idx += 1

        self.assign_layers()

    def mutate_add_connection(self):
        """Tries to add a new connection between two previously unconnected nodes."""
        adjacency_map = {nid: set() for nid in self.nodes}
        for conn in self.connections.values():
            if conn.enabled:
                adjacency_map[conn.in_node].add(conn.out_node)

        possible_starts = [n.id for n in self.nodes.values() if n.type != "output"]
        possible_ends = [n.id for n in self.nodes.values() if n.type != "input"]

        # Find pairs of unconnected nodes
        candidate_pairs = [
            (start, end) 
            for start in possible_starts 
            for end in possible_ends
            if end not in adjacency_map[start] and self.nodes[start].layer < self.nodes[end].layer
        ]

        if not candidate_pairs:
            return

        # Pick a random pair
        start_node_id, end_node_id = random.choice(candidate_pairs)
        self.connections[self.connection_idx] = ConnectionGene(self.connection_idx, start_node_id, end_node_id)
        self.connection_idx += 1

    def mutate_remove_connection(self):
        """Tries to remove a random connection."""
        if not self.connections:
            return

        opp_adjacency_map = {nid: set() for nid in self.nodes}
        for conn in self.connections.values():
            if conn.enabled:
                opp_adjacency_map[conn.out_node].add(conn.in_node)
        
        exclude_out_nodes = [
            n for n in opp_adjacency_map if len(opp_adjacency_map[n]) == 1 and self.nodes[n].type == "output"]
        possible_connections = [c for c in self.connections.values() if c.enabled and c.out_node not in exclude_out_nodes]

        if not possible_connections:
            return

        conn_to_remove = random.choice(possible_connections)
        conn_to_remove.enabled = False
        self.connections.pop(conn_to_remove.id)

    def mutate_add_node(self):
        """Splits an existing connection by adding a new node."""
        if not self.connections:
            return

        enabled_connections = [c for c in self.connections.values() if c.enabled]
        if not enabled_connections:
            return
            
        conn_to_split = random.choice(enabled_connections)
        conn_to_split.enabled = False
        self.connections.pop(conn_to_split.id)

        self.nodes[self.node_idx] = NodeGene(self.node_idx, node_type="hidden")
        self.node_idx += 1

        # Create two new connections
        in_node = conn_to_split.in_node
        out_node = conn_to_split.out_node
        original_weight = conn_to_split.weight

        # Connection 1: original input -> new node (weight 1.0)
        conn1 = ConnectionGene(self.connection_idx, in_node, self.node_idx - 1, weight=1.0)
        self.connections[self.connection_idx] = conn1
        self.connection_idx += 1

        # Connection 2: new node -> original output (original weight)
        conn2 = ConnectionGene(self.node_idx - 1, out_node, self.connection_idx, weight=original_weight)
        self.connections[self.connection_idx] = conn2
        self.connection_idx += 1

    def mutate_remove_node(self):
        """Tries to remove a random node."""
        hidden_nodes = [nid for nid, n in self.nodes.items() if n.type == "hidden"]
        if not hidden_nodes:
            return

        node_to_remove = random.choice(hidden_nodes)
        incoming = [c for c in self.connections.values() if c.out_node == node_to_remove]
        outgoing = [c for c in self.connections.values() if c.in_node == node_to_remove]

        # Repair connections using bypasses
        for in_conn in incoming:
            for out_conn in outgoing:
                # Check if the connection already exists
                exists = any(
                    c.in_node == in_conn.in_node and c.out_node == out_conn.out_node
                    for c in self.connections.values()
                )
                if exists:
                    continue

                new_weight = in_conn.weight * out_conn.weight
                self.connections[self.connection_idx] = ConnectionGene(
                    id=self.connection_idx,
                    in_node=in_conn.in_node,
                    out_node=out_conn.out_node,
                    weight=new_weight,
                    enabled=True
                )
                self.connection_idx += 1

        self.nodes.pop(node_to_remove)
        for conn in incoming + outgoing:
            self.connections.pop(conn.id, None)

    # # --- crossover ---
    # @staticmethod
    # def crossover(parent1: 'Genome', parent2: 'Genome', child_id: int, innovation_tracker: InnovationTracker) -> 'Genome':
    #     """Performs crossover between two parent genomes."""
    #     # Ensure parent1 is the more fit parent
    #     if parent2.fitness > parent1.fitness:
    #         parent1, parent2 = parent2, parent1

    #     child = Genome(child_id, parent1.input_size, parent1.output_size, innovation_tracker)
    #     child.nodes = copy.deepcopy(parent1.nodes)

    #     # Inherit connections
    #     innovs1 = sorted(parent1.connections.keys())
    #     innovs2 = sorted(parent2.connections.keys())
        
    #     i1, i2 = 0, 0
    #     while i1 < len(innovs1) and i2 < len(innovs2):
    #         innov1, innov2 = innovs1[i1], innovs2[i2]
    #         conn1 = parent1.connections[innov1]
    #         conn2 = parent2.connections[innov2]

    #         if innov1 == innov2: # Matching gene
    #             child.connections[innov1] = copy.deepcopy(random.choice([conn1, conn2]))
    #             i1 += 1
    #             i2 += 1
    #         elif innov1 < innov2: # Disjoint/Excess gene from parent1
    #             child.connections[innov1] = copy.deepcopy(conn1)
    #             i1 += 1
    #         else: # Disjoint gene from parent2
    #             i2 += 1

    #     # Inherit remaining excess genes from parent1
    #     while i1 < len(innovs1):
    #         innov1 = innovs1[i1]
    #         conn1 = parent1.connections[innov1]
    #         child.connections[innov1] = copy.deepcopy(conn1)
    #         i1 += 1
            
    #     return child
    
    # --- utils ---
    def assign_layers(self):
        """Assigns layers to nodes based on the number of incoming connections."""
        layers = {nid: 0 for nid, n in self.nodes.items() if n.type == "input"}
        changed = True
        while changed:
            changed = False
            for conn in self.connections.values():
                if not conn.enabled:
                    continue
                in_layer = layers.get(conn.in_node, 0)
                out_layer = layers.get(conn.out_node, 0)
                if out_layer <= in_layer:
                    layers[conn.out_node] = in_layer + 1
                    changed = True
        
        # Assign layers to node objects
        for nid, layer in layers.items():
            self.nodes[nid].layer = layer