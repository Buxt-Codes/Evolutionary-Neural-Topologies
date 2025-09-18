from typing import Optional, Dict
from dataclasses import field
import time

from ..genome import Genome

class GenomeEntry:
    id: str
    genome: Genome

    parent_id: Optional[str] = None
    generation: int = 0
    timestamp = float = field(default_factory=time.time)
    iteration_found: int = 0

    metrics: Dict[str, float] = field(default_factory=dict)