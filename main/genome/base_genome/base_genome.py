import random
import copy
from typing import Dict

from .genes import ConnectionGene, NodeGene
from ..innovation_tracker import InnovationTracker
from ...config import Config

class BaseGenome:
    """Represents an individual's genetic makeup, defining a neural network."""
    def __init__(self, id: int, input_size: int, output_size: int, innovation_tracker: InnovationTracker, config: Config):
        self.id = id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness = 0.0
        self.input_size = input_size
        self.output_size = output_size

        # Create input and output nodes
        for i in range(input_size):
            node_id = -(i + 1)
            self.nodes[node_id] = NodeGene(node_id, node_type="input")
            self.nodes[node_id].layer = 0

        for i in range(output_size):
            node_id = i
            self.nodes[node_id] = NodeGene(node_id, node_type="output")
            self.nodes[node_id].layer = 1

        self.input_nodes = [n for n in self.nodes if self.nodes[n].type == "input"]
        self.output_nodes = [n for n in self.nodes if self.nodes[n].type == "output"]

        # Create initial connections        
        for out_node in self.output_nodes:
            in_node = random.choice(self.input_nodes)
            innov = innovation_tracker.get_innovation(in_node, out_node)
            self.connections[innov] = ConnectionGene(
                in_node=in_node,
                out_node=out_node,
                innovation=innov,
                enabled=True
            )
        
        # Create remaining random connections
        for in_node in self.input_nodes:
            for out_node in self.output_nodes:
                if any(c.in_node == in_node and c.out_node == out_node for c in self.connections.values()):
                    continue
                if random.random() < config.initial_connection_prob:
                    innov = innovation_tracker.get_innovation(in_node, out_node)
                    self.connections[innov] = ConnectionGene(
                        in_node=in_node,
                        out_node=out_node,
                        innovation=innov,
                        enabled=True
                    )

    def mutate_add_connection(self, innovation_tracker: InnovationTracker):
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
        innov = innovation_tracker.get_innovation(start_node_id, end_node_id)
        self.connections[innov] = ConnectionGene(start_node_id, end_node_id, innov)

    def mutate_remove_connection(self):
        """Tries to remove a random connection."""
        if not self.connections:
            return

        opp_adjacency_map = {nid: set() for nid in self.nodes}
        for conn in self.connections.values():
            if conn.enabled:
                opp_adjacency_map[conn.out_node].add(conn.in_node)
        
        exclude_out_nodes = [n for n in opp_adjacency_map if len(opp_adjacency_map[n]) == 1 and self.nodes[n].type == "output"]
        possible_connections = [c for c in self.connections.values() if c.enabled and c.out_node not in exclude_out_nodes]

        if not possible_connections:
            return

        conn_to_remove = random.choice(possible_connections)
        conn_to_remove.enabled = False
        self.connections.pop(conn_to_remove.innovation)

    def mutate_add_node(self, innovation_tracker: InnovationTracker, global_node_id_counter: Dict):
        """Splits an existing connection by adding a new node."""
        if not self.connections:
            return

        enabled_connections = [c for c in self.connections.values() if c.enabled]
        if not enabled_connections:
            return
            
        conn_to_split = random.choice(enabled_connections)
        conn_to_split.enabled = False
        self.connections.pop(conn_to_split.innovation)

        new_node_id = global_node_id_counter['id']
        self.nodes[new_node_id] = NodeGene(new_node_id, node_type="hidden")
        global_node_id_counter['id'] += 1

        # Create two new connections
        in_node = conn_to_split.in_node
        out_node = conn_to_split.out_node
        original_weight = conn_to_split.weight

        # Connection 1: original input -> new node (weight 1.0)
        innov1 = innovation_tracker.get_innovation(in_node, new_node_id)
        conn1 = ConnectionGene(in_node, new_node_id, innov1, weight=1.0)
        self.connections[innov1] = conn1

        # Connection 2: new node -> original output (original weight)
        innov2 = innovation_tracker.get_innovation(new_node_id, out_node)
        conn2 = ConnectionGene(new_node_id, out_node, innov2, weight=original_weight)
        self.connections[innov2] = conn2

    def mutate_remove_node(self, innovation_tracker: InnovationTracker):
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
                innov = innovation_tracker.get_innovation(in_conn.in_node, out_conn.out_node)
                self.connections[innov] = ConnectionGene(
                    in_node=in_conn.in_node,
                    out_node=out_conn.out_node,
                    innovation=innov,
                    weight=new_weight,
                    enabled=True
                )

        self.nodes.pop(node_to_remove)
        for conn in incoming + outgoing:
            self.connections.pop(conn.innovation, None)

    def mutate_weights(self, config: Config):
        """Perturbs or replaces the weights of connections."""
        for conn in self.connections.values():
            if random.random() < config.weight_replace_prob:
                conn.weight = random.uniform(-1.0, 1.0)
            else:
                perturbation = random.gauss(0, config.weight_mutate_power)
                conn.weight += perturbation
                conn.weight = max(-1.0, min(1.0, conn.weight)) # Clamp weight

    def mutate_activations(self, activation_mutate_prob: float = 0.1, weights=None):
        """Perturbs the activations of nodes."""
        activations = ['relu', 'sigmoid', 'tanh']
        for node in self.nodes.values():
            if random.random() < activation_mutate_prob:
                node.activation = random.choices(activations, weights=weights)[0]

    # --- crossover ---
    @staticmethod
    def crossover(parent1: 'Genome', parent2: 'Genome', child_id: int, innovation_tracker: InnovationTracker) -> 'Genome':
        """Performs crossover between two parent genomes."""
        # Ensure parent1 is the more fit parent
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1

        child = Genome(child_id, parent1.input_size, parent1.output_size, innovation_tracker)
        child.nodes = copy.deepcopy(parent1.nodes)

        # Inherit connections
        innovs1 = sorted(parent1.connections.keys())
        innovs2 = sorted(parent2.connections.keys())
        
        i1, i2 = 0, 0
        while i1 < len(innovs1) and i2 < len(innovs2):
            innov1, innov2 = innovs1[i1], innovs2[i2]
            conn1 = parent1.connections[innov1]
            conn2 = parent2.connections[innov2]

            if innov1 == innov2: # Matching gene
                child.connections[innov1] = copy.deepcopy(random.choice([conn1, conn2]))
                i1 += 1
                i2 += 1
            elif innov1 < innov2: # Disjoint/Excess gene from parent1
                child.connections[innov1] = copy.deepcopy(conn1)
                i1 += 1
            else: # Disjoint gene from parent2
                i2 += 1

        # Inherit remaining excess genes from parent1
        while i1 < len(innovs1):
            innov1 = innovs1[i1]
            conn1 = parent1.connections[innov1]
            child.connections[innov1] = copy.deepcopy(conn1)
            i1 += 1
            
        return child
    
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