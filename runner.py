# ============================================================================
# Runner
# ============================================================================
import os
import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch
import botorch

from core import ProblemFactory
from strategies import BOStrategyFactory
from utils import BODataset, ExperimentLogger


class BORunner:
    def __init__(self, config: SimpleNamespace, local_dir: Path) -> None:
        self.config = config
        self.local_dir = local_dir
        self.logger = ExperimentLogger(config, local_dir)
        self.problem = ProblemFactory.create(**config.problem)
        self.dataset = BODataset(problem=self.problem)
        self.strategy = BOStrategyFactory.create(
            **config.strategy,
            config=config,
        )
        
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        botorch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def run(self) -> None:
        
        self.logger.info("Generating initial data ...")
        init_X, init_Y = self.strategy.generate_initial_data(self.dataset)
        self.dataset.add(init_X, init_Y, metadata={"init": True})
        self.logger.info_obs(self.dataset.data)

        for it in range(self.config.n_iter):
            self.logger.info("")
            self.logger.info(
                f"<<< Iteration {it+1:3d}/{self.config.n_iter} >>>"
            )
            
            self.logger.info("Fitting the model ...")
            model = self.strategy.fit_model(self.dataset)
            self.logger.info_model(model)

            self.logger.info("Acquiring next observation point ...")
            new_X, new_Y = self.strategy.get_next_data(
                dataset=self.dataset,
                model=model,
            )
            self.dataset.add(new_X, new_Y, metadata={"init": False})
            self.logger.info_obs(self.dataset.data[-1:])
        
        self.logger.info("Saving results ...")
        path = self.local_dir / "dataset.pth"
        self.dataset.save(path)
        self.logger.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--expidx", "-i", type=int, required=True)
    args = parser.parse_args()

    config_dir = Path("experiments/configs")
    config_path = config_dir / f"{args.config}.yaml"
    with open(config_path, mode="r") as f:
        cfg = yaml.safe_load(f)
    global_cfg = cfg["global"]
    local_cfg = cfg["local"][args.expidx]
    config = SimpleNamespace({**global_cfg, **local_cfg})
    result_dir = Path("experiments/results")
    local_dir = result_dir / args.config / config.name / str(config.seed)
    os.makedirs(local_dir, exist_ok=True)

    ###
    
    runner = BORunner(config=config, local_dir=local_dir)
    runner.run()