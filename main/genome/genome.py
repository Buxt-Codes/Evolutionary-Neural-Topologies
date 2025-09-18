from .base_genome import BaseGenome, ConnectionGene, NodeGene
from .genome_net import GenomeNet
from ..config import Config

class Genome:
    def __init__(self, config: Config):
        