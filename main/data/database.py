from typing import Dict, Optional, List
import torch
import pickle

from ..genome import Genome
from .genome_entry import GenomeEntry
from ..config import Config

class GenomeDatabase:
    def __init__(self, config: Config, path: Optional[str] = None):
        self.genomes: Dict[int, GenomeEntry] = {}
        self.config = config

        self.current_iteration: int = 0
        self.current_idx: int = 0

        self.islands: List[Dict[int, GenomeEntry]] = [{} for _ in range(config.num_islands)]
        self.best_program_per_island: List[GenomeEntry] = [None for _ in range(config.num_islands)]
        self.current_island: int = 0

        self.best_program: GenomeEntry = None
        self.current_program: GenomeEntry = None

        if path is not None:
            self.load(path)
        else:
            self.initialise()
    
    def load(self, path: str):
        with open(path, "rb") as f:
            load = pickle.load(f)
        
        self.genomes = load["genomes"]
        self.islands = load["islands"]
        self.best_program_per_island = load["best_program_per_island"]
        self.current_iteration = load["current_iteration"]
        self.current_idx = load["current_idx"]
        self.current_island = load["current_island"]
        self.best_program = load["best_program"]
        self.current_program = load["current_program"]
    
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "genomes": self.genomes,
                "islands": self.islands,
                "best_program_per_island": self.best_program_per_island,
                "current_iteration": self.current_iteration,
                "current_idx": self.current_idx,
                "current_island": self.current_island,
                "best_program": self.best_program,
                "current_program": self.current_program
            }, f)

    def initialise(self):
        for i in range(self.config.num_islands):
            for _ in range(self.config.num_programs_per_island):
                genome = Genome(self.current_idx, self.config)
                self.add_genome(genome, island=i, generation=0, iteration_found=0, metrics={})
                self.current_idx += 1
    
    def add_genome(
        self, 
        genome: Genome,
        island: Optional[int] = None,
        parent_id: Optional[int] = None,
        generation: int = 0, 
        iteration_found: int = 0, 
        metrics: Dict[str, float] = {}
    ):
        if island is None:
            if parent_id in self.genomes:
                island = self.genomes[parent_id].island
            else:
                island = self.current_island

        entry = GenomeEntry(
            id=self.current_idx, 
            genome=genome, 
            island=island, 
            parent_id=parent_id, 
            generation=generation, 
            iteration_found=iteration_found, 
            metrics=metrics
        )

        num_nodes = entry.genome.num_nodes
        num_layers = entry.genome.num_layers

        nodes_bin = self._get_bin(num_nodes, max_value=self.config.max_nodes_bin, num_bins=self.config.num_nodes_bins)
        layers_bin = self._get_bin(num_layers, max_value=self.config.max_layers_bin, num_bins=self.config.num_layers_bins)


        self.genomes[genome.id] = entry
        self.islands[island][genome.id] = entry

    def _get_bin(self, value: float, max_value: int = 1000, num_bins: int = 10) -> int:
        if num_bins == 1:
            return 0
        elif num_bins < 1: 
            raise ValueError("num_bins must be > 1")
    
        # Calculate the base for the max_value to be mapped to the last bin
        base = (max_value + 1) ** (1 / (num_bins - 1))

        # Map value to bin
        val_t = torch.tensor(value)
        bin_idx = torch.floor(torch.log(val_t + 1) / torch.log(torch.tensor(base)))

        # Clamp to valid range
        bin_idx = torch.clamp(value, min=0, max=num_bins - 1)

        return int(bin_idx.item())

    def _get_coords(self, ):
        

                        
