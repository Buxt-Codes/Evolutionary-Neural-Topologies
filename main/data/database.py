from typing import Dict, Optional, List
import math
import pickle
import uuid
import random
import copy
import os

from ..genome import Genome
from .genome_entry import GenomeEntry
from ..config import Config

class GenomeDatabase:
    def __init__(self, config: Config, path: Optional[str] = None):
        self.genomes: Dict[str, GenomeEntry] = {}
        self.config = config

        self.islands: List[Dict[str, GenomeEntry]] = [{} for _ in range(config.num_islands)]
        self.best_genome_per_island: List[Optional[GenomeEntry]] = [None for _ in range(config.num_islands)]
        self.current_island: int = 0

        self.best_genome: Optional[GenomeEntry] = None 
        self.best_genome_id: Optional[str] = None

        if path is not None:
            self.load(path)
        else:
            self.initialise()
    
    def load(self, path: str):
        with open(path, "rb") as f:
            load = pickle.load(f)
        
        self.genomes = load["genomes"]
        self.islands = load["islands"]
        self.best_genome_per_island = load["best_genome_per_island"]
        self.current_island = load["current_island"]
        self.best_genome = load["best_genome"]
    
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "genomes": self.genomes,
                "islands": self.islands,
                "best_genome_per_island": self.best_genome_per_island,
                "current_island": self.current_island,
                "best_genome": self.best_genome,
            }, f)

    def initialise(self):
        for i in range(self.config.num_islands):
            for _ in range(self.config.initial_pop_per_island):
                genome = GenomeEntry(id=str(uuid.uuid4()), genome=Genome(self.config), island=i, generation=0, iteration_found=0, metrics={})
                self.add(genome)
    
    def add(
        self,
        genome: GenomeEntry, 
    ):

        num_nodes = genome.genome.num_nodes
        num_layers = genome.genome.num_layers

        nodes_bin = self._get_bin(num_nodes, max_value=self.config.max_nodes_bin, num_bins=self.config.num_nodes_bins)
        layers_bin = self._get_bin(num_layers, max_value=self.config.max_layers_bin, num_bins=self.config.num_layers_bins)

        genome.coords = self._get_coords(nodes_bin, layers_bin)

        existing_genome = self.get_genome_by_coords(genome.island, genome.coords)
        if existing_genome is not None:
            if self._is_better(genome, existing_genome):
                self._delete_genome(existing_genome.id)
        
        self.genomes[genome.id] = genome
        self.islands[genome.island][genome.coords] = genome

        self._enforce_population_limit(genome.id)
        self._update_best_genome_per_island(genome)
        self._update_best_genome(genome)
    
    def sample(self, island: Optional[int] = None, exclude_genome: Optional[GenomeEntry] = None, exploitation: bool = False) -> GenomeEntry:
        if island is None:
            island = self.current_island

        if len(self.islands[island]) == 0:
            return random.choice(list(self.genomes.values()))

        if exclude_genome is not None:
            if exclude_genome.coords in self.islands[island]:
                genomes = [x for x in self.islands[island].values() if x.id != exclude_genome.id]
                if len(genomes) == 0:
                    return random.choice(list(self.genomes.values()))
            else:
                genomes = list(self.islands[island].values())
        else:
            genomes = list(self.islands[island].values())

        if exploitation:
            return min(genomes, key=lambda x: x.metrics.get("loss", float("inf")))

        if len(genomes) == 1:
            return genomes[0]

        return random.choice(genomes)

    def rotate_island(self):
        self.current_island = (self.current_island + 1) % self.config.num_islands
    
    def migrate_genomes(self):
        for i in range(self.config.num_islands):
            island_genomes = sorted(self.islands[i].values(), key=lambda x: x.metrics.get("loss", float("inf")))
            num_migrating = min(len(island_genomes), self.config.num_to_migrate)
            for genome in island_genomes[:num_migrating]:
                island = (i + 1) % self.config.num_islands
                migrating_genome = copy.deepcopy(genome)
                migrating_genome.island = island
                self.add(migrating_genome)

    def _delete_genome(self, id: str):
        if id in self.genomes:
            genome = self.genomes[id]
            if genome.coords in self.islands[genome.island]:
                del self.islands[genome.island][genome.coords]
            
            del self.genomes[id]
        
        file_path = os.path.join(self.config.model_path, f"{id}.pt")
        if os.path.exists(file_path):
            os.remove(file_path)
        
    def _get_bin(self, value: float, max_value: int = 1000, num_bins: int = 10) -> int:
        if num_bins == 1:
            return 0
        elif num_bins < 1: 
            raise ValueError("num_bins must be > 1")
    
        value = max(0.0, value)  

        # Calculate base so last bin represents max_value
        base = (max_value + 1) ** (1 / (num_bins - 1))

        # Map value to bin
        bin_idx = math.floor(math.log(value + 1) / math.log(base))

        # Clamp to valid range
        bin_idx = max(0, min(bin_idx, num_bins - 1))

        return int(bin_idx)

    def _get_coords(self, nodes_bin: int, layers_bin: int) -> str:
        return f"{nodes_bin}_{layers_bin}"

    def get_genome(self, id: str) -> Optional[GenomeEntry]:
        return self.genomes.get(id)
    
    def get_genome_by_coords(self, island: int, coords: str) -> Optional[GenomeEntry]:
        return self.islands[island].get(coords)
    
    def _is_better(self, genome1: GenomeEntry, genome2: GenomeEntry) -> bool:
        loss1 = genome1.metrics.get("loss")
        loss2 = genome2.metrics.get("loss")

        if loss1 is not None and loss2 is not None:
            return loss1 < loss2
        elif loss1 is not None:
            return True
        return False

    def _enforce_population_limit(self, exclude_genome_id: str):
        if len(self.genomes) <= self.config.max_population:
            return
        
        num_to_remove = len(self.genomes) - self.config.max_population

        all_genomes = list(self.genomes.values())
        sorted_genomes = sorted(
            all_genomes,
            key=lambda x: x.metrics.get("loss", float("inf"))
        )

        genomes_to_remove = []
        protected_genomes = {exclude_genome_id, self.best_genome_id} - {None}

        for genome in sorted_genomes:
            if len(genomes_to_remove) >= num_to_remove:
                break

            if genome.id not in protected_genomes:
                genomes_to_remove.append(genome)
        
        for genome in genomes_to_remove:
            if genome.id in self.genomes:
                del self.genomes[genome.id]
            
            if genome.coords in self.islands[genome.island]:
                del self.islands[genome.island][genome.coords]
    
    def _update_best_genome(self, genome: GenomeEntry):
        if self.best_genome is None:
            self.best_genome = genome
            self.best_genome_id = genome.id
            return
        
        best_genome = self.best_genome
        if self._is_better(genome, best_genome):
            self.best_genome = genome
            self.best_genome_id = genome.id
    
    def _update_best_genome_per_island(self, genome: GenomeEntry):
        island = genome.island
        if self.best_genome_per_island[island] is None:
            self.best_genome_per_island[island] = genome
            return
        
        best_genome_for_island = self.best_genome_per_island[island]
        if best_genome_for_island is None:
            self.best_genome_per_island[island] = genome
            return
        if self._is_better(genome, best_genome_for_island):
            self.best_genome_per_island[island] = genome
        