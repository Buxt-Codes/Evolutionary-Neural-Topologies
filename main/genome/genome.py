

from .base_genome import BaseGenome
from ..config import Config

class Genome:
    def __init__(self, id: int, config: Config):
        self.id: int = id
        self.genome = BaseGenome(id, config)
        self.fitness: float = 0.0
        self.config = config

    def mutate(self):
        