# ============================================================================
# Logger
# ============================================================================
import os
import time
import logging
from pathlib import Path
from typing import overload
from types import SimpleNamespace

import yaml
from botorch.models.model import Model

from utils.data import ObservedData


class ExperimentLogger:
    def __init__(self, config: SimpleNamespace, local_dir: Path) -> None:
        self.config = config
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file = self.local_dir / "experiment.log"
        file_handler = logging.FileHandler(file, mode="w")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        self.logger.addHandler(console_handler)

        self.start()
        self.save_config(config)

    def start(self) -> None:
        self.logger.info("============================================")
        self.logger.info("    ____  ____     _____ __             __ ")
        self.logger.info("   / __ )/ __ \\   / ___// /_____ ______/ /_")
        self.logger.info("  / __  / / / /   \\__ \\/ __/ __ `/ ___/ __/")
        self.logger.info(" / /_/ / /_/ /   ___/ / /_/ /_/ / /  / /_  ")
        self.logger.info("/_____/\\____/   /____/\\__/\\__,_/_/   \\__/  ")
        self.logger.info("")
        self.logger.info("+------------------------------------------+")
        self.logger.info(f"+ CONFIG : {self.local_dir.parent.parent.stem}")
        self.logger.info(f"+ EXPERIMENT : {self.local_dir.parent.stem}")
        self.logger.info(f"+ SEED : {self.local_dir.stem}")
        self.logger.info("+------------------------------------------+")
        self.start_time = time.perf_counter()

    def end(self) -> None:
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        self.logger.info("+------------------------------------------+")
        self.logger.info(f"+ CONFIG : {self.local_dir.parent.parent.stem}")
        self.logger.info(f"+ EXPERIMENT : {self.local_dir.parent.stem}")
        self.logger.info(f"+ SEED : {self.local_dir.stem}")
        self.logger.info("+------------------------------------------+")
        self.logger.info(f"+ TIME : {elapsed} s")
        self.logger.info("+------------------------------------------+")
        self.logger.info("    ____  ____     _______       _      __  ")
        self.logger.info("   / __ )/ __ \\   / ____(_)___  (_)____/ /_ ")
        self.logger.info("  / __  / / / /  / /_  / / __ \\/ / ___/ __ \\")
        self.logger.info(" / /_/ / /_/ /  / __/ / / / / / (__  ) / / /")
        self.logger.info("/_____/\\____/  /_/   /_/_/ /_/_/____/_/ /_/ ")
        self.logger.info("")
        self.logger.info("============================================")

    def save_config(self, config: SimpleNamespace) -> None:
        config_path = self.local_dir / "config.yaml"
        with open(config_path, mode="w") as f:
            yaml.safe_dump(vars(config), f, sort_keys=False)
        self.logger.info(f"Configuration is saved to {config_path}")

    def info_obs(self, obs_list: list[ObservedData]) -> None:
        self.logger.info("+------------------------------------------+")
        for obs in obs_list:
            self.logger.info(f"+ Observation")
            self.logger.info(f"+   X : {obs.X}")
            self.logger.info(f"+   Y : {obs.Y}")
            self.logger.info(f"+   timestamp : {obs.timestamp}")
        self.logger.info("+------------------------------------------+")

    def info_model(self, model: Model) -> None:
        self.logger.info("+------------------------------------------+")
        for name, param in model.named_parameters():
            self.logger.info(f"+ {name} : {param.data}")
        self.logger.info("+------------------------------------------+")

    def info(self, message: str) -> None:
        self.logger.info(message)

    def __del__(self) -> None:
        self.end()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)