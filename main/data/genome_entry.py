from typing import Optional, Dict
from dataclasses import dataclass, field
import time

from ..genome import Genome

@dataclass
class GenomeEntry:
    id: str
    genome: Genome
    island: int
    
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0

    coords: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)