import random

from .base_genome import BaseGenome
from.genome_net import GenotypeNet
from ..config import Config

class Genome:
    def __init__(self, config: Config):
        self.genome = BaseGenome(config)
        self.fitness: float = 0.0
        
        self.update_metadata()

    def update_metadata(self):
        self.num_layers = max(self.genome.nodes.values(), key=lambda n: n.layer).layer
        self.num_nodes = len(self.genome.nodes)

    def mutate(self):
        config = self.genome.config

        mutations = 0
        while mutations < (config.max_mutations if config.max_mutations > 0 else 1):
            prob = random.random()
            if prob < config.prob_add_connection:
                self.genome.mutate_add_connection()
                mutations += 1
            elif prob < config.prob_add_node + config.prob_add_connection:
                self.genome.mutate_add_node()
                mutations += 1
            elif prob < config.prob_remove_connection + config.prob_add_node + config.prob_add_connection:
                self.genome.mutate_remove_connection()
                mutations += 1
            elif prob < config.prob_remove_node + config.prob_remove_connection + config.prob_add_node + config.prob_add_connection:
                self.genome.mutate_remove_node()
                mutations += 1
            
        self.update_metadata()
    
    def crossover(self, other: 'Genome'):
        self.genome = BaseGenome.crossover(self.genome, other.genome, self.genome.config)
        self.update_metadata()

    def build_net(self):
        return GenotypeNet(self.genome, self.genome.config)

    def update_genome(self, genome: BaseGenome):
        self.genome = genome
        return self