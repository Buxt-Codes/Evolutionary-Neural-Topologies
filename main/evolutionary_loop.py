import asyncio
import multiprocessing as mp
from typing import Optional, Dict, List, Tuple
import random
import uuid
import copy
import os
from time import monotonic
import traceback
import pandas as pd

import torch
import logging

from .config import Config
from .genome import Genome
from .data import GenomeEntry, GenomeDatabase
from .evaluate import evaluate

# -------------------------------
# Logging helpers
# -------------------------------
class QueueHandler(logging.Handler):
    def __init__(self, queue: mp.Queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        try:
            self.queue.put(record)
        except Exception:
            pass

def listener_process(queue: mp.Queue, log_file: str):
    logger = logging.getLogger("GeneticEvolution")
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    fmt = logging.Formatter("[%(asctime)s][%(processName)s][%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    while True:
        record = queue.get()
        if record is None:
            break
        logger.handle(record)

# -------------------------------
# Worker logic
# -------------------------------

def _worker_target(
    config: Config, 
    genome: GenomeEntry,
    inspiration: GenomeEntry,
    iteration: int,
    q: mp.Queue,
    env: dict,
    log_queue: mp.Queue,
    gpu_mem_fraction: float = 0.5
) -> Optional[GenomeEntry]:
    import os

    os.environ.update(env)

    logger = logging.getLogger(f"Worker-{iteration}")
    logger.setLevel(logging.DEBUG)
    if log_queue:
        logger.addHandler(QueueHandler(log_queue))

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)

    try:
        from .evaluate import evaluate
        from .data import GenomeEntry

        child_id = str(uuid.uuid4())
        child_genome = copy.deepcopy(genome.genome)
        inspiration_genome = copy.deepcopy(inspiration.genome)

        if random.random() < config.crossover_prob:
            child_genome.crossover(inspiration_genome)
        else:
            child_genome.mutate()

        metrics = evaluate(child_id, child_genome, config)

        result = GenomeEntry(
            id=child_id,
            genome=child_genome,
            island=genome.island,
            parent_id=inspiration.id,
            generation=genome.generation + 1,
            iteration_found=iteration,
            metrics=metrics,
        )
        q.put(result)

    except Exception as e:
        logger.error(f"[Worker Error]Iteration {iteration}: {traceback.format_exc()}")
        q.put(None)

class EvolutionaryLoop:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.timeout = getattr(config, "worker_timeout", 600)  
        self.max_workers = config.num_workers
        self.gpu_mem_fraction = getattr(config, "worker_gpu_fraction", 0.5)
        self.log_path = getattr(config, "log_oath", "evolution.log")

    async def run_evolution(
        self,
        from_checkpoint: bool = False
    ) -> None:
        self.db = GenomeDatabase(self.config, path=self.config.db_path if from_checkpoint else None)
        
        os.makedirs(self.config.model_path, exist_ok=True)
        
        if not os.path.exists(self.config.stats_path):
            headers = ["id", "generation", "iteration", "parent", "loss", "rmse"]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.config.stats_path, index=False)

        self.log_queue = mp.Queue()
        listener = mp.Process(target=listener_process, args=(self.log_queue, self.log_path))
        listener.start()

        logger = logging.getLogger("GeneticEvolution")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(QueueHandler(self.log_queue))

        logger.info("Starting Genetic Evolution Loop")

        pending: Dict[int, Tuple[mp.Process, mp.Queue, float]] = {}
        current_iteration = 0
        current_island = 0
        completed_iterations = 0

        loop = asyncio.get_event_loop()

        while len(pending) < self.max_workers and current_iteration < self.config.max_iterations:
            proc, q = self._spawn_worker(current_iteration, current_island)
            pending[current_iteration] = (proc, q, monotonic())
            logger.info(f"Submitted Worker: {current_iteration}")
            current_iteration += 1
            current_island = (current_island + 1) % self.config.num_islands
        
        
        while pending and completed_iterations < self.config.max_iterations:
            done_iters = []

            for iteration, (proc, q, start_time) in list(pending.items()):
                # check timeout
                if monotonic() - start_time > self.timeout:
                    logger.warning(f"Iteration {iteration}: Exceeded {self.timeout}s. Killing process {proc.pid}.")
                    proc.terminate()
                    proc.join()
                    # free GPU cache on main process
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    done_iters.append(iteration)
                    continue
                
                # Non-blocking result
                try:
                    result = q.get_nowait()
                    logger.info(f"Iteration {iteration}: Received, metrics={result.metrics}")
                except Exception:
                    continue
                else:
                    proc.join()
                    if result is not None:
                        self.db.add(result)
                        self.log_stats(result)
                        if current_iteration % self.config.db_save_interval == 0:
                            self.db.save(self.config.db_path)
                            self.log_checkpoint(completed_iterations)
                        completed_iterations += 1
                        logger.info(f"Iteration {iteration}: Saved, metrics={result.metrics}")
                    done_iters.append(iteration)

            # cleanup finished/terminated
            for it in done_iters:
                pending.pop(it, None)

            # spawn new workers if slots are free
            while len(pending) < self.max_workers and current_iteration < self.config.max_iterations:
                proc, q = self._spawn_worker(current_iteration, current_island)
                pending[current_iteration] = (proc, q, monotonic())
                logger.info(f"Submitted Worker: {current_iteration}")
                current_iteration += 1
                current_island = (current_island + 1) % self.config.num_islands

            await asyncio.sleep(0.01)

        self.log_queue.put(None)
        listener.join()
        logger.info("Genetic Evolution Loop Complete")

    def _spawn_worker(self, iteration: int, island: int) -> Tuple[mp.Process, mp.Queue]:
        parent_genome = self.db.sample(island=island, exploitation=False)
        inspiration = self.db.sample(
            island=island,
            exclude_genome=parent_genome,
            exploitation=True if random.random() < self.config.exploitation_prob else False,
        )
        q = mp.Queue()
        proc = mp.Process(
            target=_worker_target,
            args=(self.config, parent_genome, inspiration, iteration, q, dict(os.environ), self.log_queue, self.gpu_mem_fraction),
        )
        proc.start()
        return proc, q

    def log_stats(self, result: GenomeEntry) -> None:
        df = pd.DataFrame(
            {
                "id": [result.id],
                "generation": [result.generation],
                "iteration": [result.iteration_found],
                "parent": [result.parent_id],
                "loss": [result.metrics["loss"]],
                "rmse": [result.metrics["rmse"]]
            }
        )
        df.to_csv(self.config.stats_path, mode="a", header=False, index=False)
    
    def log_checkpoint(self, iteration: int):
        best = self.db.best_genome
        best_islands = sorted(
            self.db.best_genome_per_island.items(),
            key=lambda x: x[1].metrics["loss"]
        )

        islands_str = " | ".join(
            f"Island: {island_id}: loss={genome.metrics['loss']:.6f}, rmse={genome.metrics['rmse']:.6f}"
            for island_id, genome in best_islands
        )

        logging.info(
            f"[Checkpoint] Iteration: {iteration} | saved to {self.config.db_path} | "
            f"Global Best -> loss={best.metrics['loss']:.6f}, rmse={best.metrics['rmse']:.6f} | "
            f"Best per Island -> {islands_str}"
        )