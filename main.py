import asyncio

from main import EvolutionaryLoop, Config

if __name__ == "__main__":
    config = Config()
    evolutionary_loop = EvolutionaryLoop(config)
    asyncio.run(evolutionary_loop.run_evolution())