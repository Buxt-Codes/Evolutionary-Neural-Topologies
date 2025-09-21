import random
import copy
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx

from .genes import ConnectionGene, NodeGene
from ...config import Config

class BaseGenome:
    """Represents an individual's genetic makeup, defining a neural network."""
    def __init__(self, config: Config):
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness = 0.0
        self.node_idx = 0
        self.connection_idx = 0
        self.config = config

        # Create input and output nodes
        for _ in range(config.input_size):
            self.nodes[self.node_idx] = NodeGene(self.node_idx, node_type="input")
            self.node_idx += 1
        
        for _ in range(config.output_size):
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
            num_layers = random.randint(2, max_layers + 2)
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
                        if self.layers[in_layer]:
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
                        if self.layers[out_layer]:
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

        self.assign_layers()

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

        self.prune()

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

        self.assign_layers()

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
        
        self.prune()

    # --- crossover ---
    @staticmethod
    def crossover(parent1: 'BaseGenome', parent2: 'BaseGenome', config: Config) -> 'BaseGenome':
        """Performs crossover between two parent genomes."""

        # Create base child
        child = copy.deepcopy(parent1)

        # Random crossover point
        start_frac, end_frac = sorted([random.random(), random.random()])

        p1_max_layer = max(parent1.nodes.values(), key=lambda n: n.layer).layer
        p2_max_layer = max(parent2.nodes.values(), key=lambda n: n.layer).layer

        # Compute layer ranges
        p1_start = max(1, round(start_frac * p1_max_layer))
        p1_end = min(p1_max_layer - 1, round(end_frac * p1_max_layer))
        p2_start = max(1, round(start_frac * p2_max_layer))
        p2_end = min(p2_max_layer - 1, round(end_frac * p2_max_layer))

        # Helper: select node IDs in a layer range
        def select_block_nodes(genome, start_layer, end_layer):
            return [n.id for n in genome.nodes.values() if start_layer <= n.layer <= end_layer]

        # Remove block from child
        child_block_nodes = select_block_nodes(child, p1_start, p1_end)
        for nid in child_block_nodes:
            child.nodes.pop(nid, None)

        # Insert block from p2
        parent2_block_nodes = select_block_nodes(parent2, p2_start, p2_end)
        id_map = {}  # parent2_node_id -> new child node_id
        for nid in parent2_block_nodes:
            node = copy.deepcopy(parent2.nodes[nid])
            new_id = child.node_idx
            child.node_idx += 1
            id_map[nid] = new_id
            # Add in placeholder layer for now
            node.layer = -1
            node.id = new_id
            child.nodes[new_id] = node
        
        # Remove existing connections from child block
        connections_to_remove = []
        for cid, conn in child.connections.items():
            if conn.in_node in child_block_nodes or conn.out_node in child_block_nodes:
                connections_to_remove.append(cid)
        for cid in connections_to_remove:
            child.connections.pop(cid)
        
        # Layer mapping p2 -> p1
        layer_map = {}
        for l2 in range(p2_max_layer + 1):
            scaled_layer = int(l2 * p1_max_layer / p2_max_layer)
            layer_map[l2] = scaled_layer

        # p1 layers
        p1_layers = {}
        for n in parent1.nodes.values():
            if p1_layers.get(n.layer) is None:
                p1_layers[n.layer] = [n]
            else:
                p1_layers[n.layer].append(n)

        # p2 layers
        p2_layers = {}
        for n in parent2.nodes.values():
            if p2_layers.get(n.layer) is None:
                p2_layers[n.layer] = [n]
            else:
                p2_layers[n.layer].append(n)

        # Add connections from p2
        for conn in parent2.connections.values():
            if conn.in_node in parent2_block_nodes and conn.out_node in parent2_block_nodes:
                new_conn = copy.deepcopy(conn)
                new_conn.in_node = id_map[conn.in_node]
                new_conn.out_node = id_map[conn.out_node]
                new_conn.id = child.connection_idx

                child.connections[child.connection_idx] = new_conn
                child.connection_idx += 1
            
            elif conn.in_node in parent2_block_nodes:
                new_conn = copy.deepcopy(conn)
                new_conn.in_node = id_map[conn.in_node]
                new_conn.id = child.connection_idx

                out_node_layer = layer_map[parent2.nodes[conn.out_node].layer]
                if out_node_layer == child.nodes[new_conn.in_node].layer:
                    out_node_layer += 1
                while out_node_layer not in p1_layers:
                    out_node_layer += 1
                    if out_node_layer > p1_max_layer:
                        break
                if out_node_layer not in p1_layers:
                    continue

                possible_out_nodes = {n.id: n.avg_activations if n.avg_activations is not None else 0 for n in p1_layers[out_node_layer]}
                target_activation = parent2.nodes[conn.out_node].avg_activations
                if target_activation:
                    closest_nid = min(
                        possible_out_nodes, 
                        key=lambda nid: abs(possible_out_nodes[nid] - target_activation)
                    
                    )
                else:
                    closest_nid = random.choice(list(possible_out_nodes))
                new_conn.out_node = closest_nid

                child.connections[child.connection_idx] = new_conn
                child.connection_idx += 1
            
            elif conn.out_node in parent2_block_nodes:
                new_conn = copy.deepcopy(conn)
                new_conn.out_node = id_map[conn.out_node]
                new_conn.id = child.connection_idx

                in_node_layer = layer_map[parent2.nodes[conn.in_node].layer]
                if in_node_layer == child.nodes[new_conn.out_node].layer:
                    in_node_layer -= 1
                while in_node_layer not in p1_layers:
                    in_node_layer -= 1
                    if in_node_layer < 0:
                        break
                if in_node_layer not in p1_layers:
                    continue
                    
                possible_in_nodes = {n.id: n.avg_activations if n.avg_activations is not None else 0 for n in p1_layers[in_node_layer]}
                target_activation = parent2.nodes[conn.in_node].avg_activations
                if target_activation:
                    closest_nid = min(
                        possible_in_nodes, 
                        key=lambda nid: abs(possible_in_nodes[nid] - target_activation)
                    )       
                else:
                    closest_nid = random.choice(list(possible_in_nodes))
                new_conn.in_node = closest_nid

                child.connections[child.connection_idx] = new_conn
                child.connection_idx += 1
        
        child.prune()
        
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
    
    def prune(self):
        """Prune disconnected nodes and their connections."""
        # Build directed graph
        G = nx.DiGraph()
        for nid in self.nodes:
            G.add_node(nid)
        for cid, conn in self.connections.items():
            if conn.enabled:
                G.add_edge(conn.in_node, conn.out_node)

        # Find nodes reachable from inputs and nodes that can reach outputs
        reachable_from_inputs = set()
        for inp in self.input_nodes:
            reachable_from_inputs |= nx.descendants(G, inp) | {inp}

        can_reach_outputs = set()
        for out in self.output_nodes:
            can_reach_outputs |= nx.ancestors(G, out) | {out}

        valid_nodes = reachable_from_inputs & can_reach_outputs

        # Remove invalid nodes
        invalid_nodes = set(G.nodes()) - valid_nodes
        G.remove_nodes_from(invalid_nodes)

        # Update genome's nodes + connections
        self.nodes = {nid: self.nodes[nid] for nid in G.nodes()}
        self.connections = {
            cid: conn for cid, conn in self.connections.items()
            if conn.in_node in G and conn.out_node in G
        }
        self.assign_layers()

    @staticmethod
    def visualize_genome(genome, ax=None):
        """
        Visualize a BaseGenome as a directed graph.
        Inputs = green, hidden = blue, outputs = red.
        Enabled edges = solid, disabled edges = dashed.
        """
        G = nx.DiGraph()

        # Add nodes
        for node_id, node in genome.nodes.items():
            if node.type == "input":
                color = "lightgreen"
            elif node.type == "output":
                color = "salmon"
            else:
                color = "lightblue"
            G.add_node(node_id, layer=node.layer, color=color)

        # Add edges
        for conn in genome.connections.values():
            style = "solid" if conn.enabled else "dashed"
            G.add_edge(conn.in_node, conn.out_node, weight=conn.weight, style=style)

        # Layout: group nodes by layer
        pos = {}
        layer_nodes = {}
        for node_id, node in genome.nodes.items():
            layer_nodes.setdefault(node.layer, []).append(node_id)

        # Space layers out
        for layer, nodes in layer_nodes.items():
            for i, n in enumerate(nodes):
                pos[n] = (layer, -i)

        # Draw nodes
        node_colors = [G.nodes[n]["color"] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)

        # Draw edges
        edge_colors = []
        styles = []
        for u, v, data in G.edges(data=True):
            edge_colors.append("black")
            styles.append(data["style"])
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, style=styles, ax=ax)

        # Draw labels
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        if ax is None:
            plt.show()