import random

from .base_genome import BaseGenome
from.genome_net import GenotypeNet
from ..config import Config

class Genome:
    def __init__(self, id: int, config: Config):
        self.id: int = id
        self.genome = BaseGenome(id, config)
        self.fitness: float = 0.0
        
        self.update_metadata()

    def update_metadata(self):
        self.num_layers = max(self.genome.nodes.values(), key=lambda n: n.layer).layer
        self.num_nodes = len(self.genome.nodes)

    def mutate(self):
        config = self.genome.config

        mutations = 0
        while mutations < (config.max_mutations if config.max_mutations > 0 else 1):
            prob_conditions = {
                "add_connection": config.prob_add_connection, 
                "remove_connection": config.prob_remove_connection, 
                "add_node": config.prob_add_node, 
                "remove_node": config.prob_remove_node
            }
            
            probs = {key: random.random() for key in prob_conditions}
            prob_order = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            mutation_funcs = {
                "add_connection": self.genome.mutate_add_connection,
                "remove_connection": self.genome.mutate_remove_connection,
                "add_node": self.genome.mutate_add_node,
                "remove_node": self.genome.mutate_remove_node,
            }

            for key, score in prob_order:
                if score < prob_conditions[key]:
                    mutation_funcs[key]()
                    mutations += 1
                    break
            
        self.update_metadata()
    
    def crossover(self, id: int, other: 'Genome'):
        self.genome = BaseGenome.crossover(self.genome, other.genome, self.id, self.genome.config)
        self.update_metadata()

    def build_net(self):
        return GenotypeNet(self.genome, self.genome.config)
    
    def reset(self):
        self.genome = BaseGenome(self.id, self.genome.config)
    
    def __repr__(self):
        return f"Genome(id={self.id}, fitness={self.fitness})"