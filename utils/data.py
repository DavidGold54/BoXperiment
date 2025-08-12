# ============================================================================
# Data
# ============================================================================
from typing import Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem


# ----------------------------------------------------------------------------
# ObservedData
# ----------------------------------------------------------------------------
@dataclass
class ObservedData:
    """A class representing a single observation data record.
    """

    X: Tensor
    Y: Tensor
    C: Tensor | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# BODataset
# ----------------------------------------------------------------------------
class BODataset:
    """A class for managing observational data obtained from BO experiments.
    """

    def __init__(self, problem: BaseTestProblem) -> None:
        self.problem = problem
        self.data: list[ObservedData] = []

    def add(
        self,
        new_X: Tensor,
        new_Y: Tensor,
        new_C: Tensor | None = None,
        metadata: dict[str, Any] = {},
    ) -> None:
        new_X = torch.atleast_2d(new_X)
        new_Y = torch.atleast_2d(new_Y)
        assert new_X.size(-1) == self.problem.dim
        if hasattr(self.problem, "num_constraints"):
            new_C = torch.atleast_2d(new_C)
            for X, Y, C in zip(new_X, new_Y, new_C):
                obs = ObservedData(X=X, Y=Y, C=C, metadata=metadata)
                self.data.append(obs)
        else:
            for X, Y in zip(new_X, new_Y):
                obs = ObservedData(X=X, Y=Y, metadata=metadata)
                self.data.append(obs)

    def get(self) -> list[tuple[Tensor, Tensor]]:
        train_data = []
        data_X = torch.stack([obs.X for obs in self.data])
        data_Y = torch.stack([obs.Y for obs in self.data])
        for i in data_Y.size(-1):
            mask = ~torch.isnan(data_Y[:, i])
            train_X = data_X[mask, :]
            train_Y = data_X[:, i:i+1]
            train_data.append((train_X, train_Y))
        if hasattr(self.problem, "num_constraints"):
            data_C = torch.stack([obs.C for obs in self.data])
            for i in data_C.size(-1):
                mask = ~torch.isnan(data_C[:, i])
                train_X = data_X[mask, :]
                train_C = data_C[:, i:i+1]
                train_data.append((train_X, train_C))
        return train_data

    def save(self, path: Path) -> None:
        data = {
            "problem": self.problem,
            "data_X": torch.stack([obs.X for obs in self.data]),
            "data_Y": torch.stack([obs.Y for obs in self.data]),
            "timestamp": [obs.timestamp for obs in self.data],
            "metadata": [obs.metadata for obs in self.data],
        }
        if hasattr(self.problem, "num_constraints"):
            data["data_C"] = torch.stack([obs.C for obs in self.data])
        torch.save(data, path)

    @classmethod
    def load(cls, path: Path) -> "BODataset":
        data = torch.load(path, weights_only=False)
        dataset = cls(problem=data["problem"])
        for i in range(len(data["data_X"])):
            if hasattr(dataset.problem, "num_constraints"):
                obs = ObservedData(
                    X=data["data_X"][i],
                    Y=data["data_Y"][i],
                    C=data["data_C"][i],
                    timestamp=data["timestamp"][i],
                    metadata=data["metadata"][i],
                )
            else:
                obs = ObservedData(
                    X=data["data_X"][i],
                    Y=data["data_Y"][i],
                    timestamp=data["timestamp"][i],
                    metadata=data["metadata"][i],
                )
            dataset.data.append(obs)
        return dataset
