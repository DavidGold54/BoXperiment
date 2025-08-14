# ============================================================================
# Base
# ============================================================================
from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from core import ModelFactory, MLLFactory, AcquisitionFactory
from utils import BODataset



# ----------------------------------------------------------------------------
# Base BO Strategy
# ----------------------------------------------------------------------------
class BaseBOStrategy(ABC):
    def __init__(
        self,
        config: SimpleNamespace
    ) -> None:
        self.config = config

    @abstractmethod
    def generate_initial_data(
        self,
        dataset: BODataset
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def fit_model(
        self,
        dataset: BODataset,
    ) -> Model:
        raise NotImplementedError

    @abstractmethod
    def get_next_data(
        self,
        dataset: BODataset,
        model: Model,
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Basic BO Strategy
# ----------------------------------------------------------------------------
class BasicBOStrategy(BaseBOStrategy):
    def generate_initial_data(
        self,
        dataset: BODataset,
    ) -> tuple[Tensor, ...]:
        init_X = draw_sobol_samples(
            bounds=torch.tensor(dataset.problem._bounds, dtype=torch.double),
            n=self.config.n_init,
            q=self.config.q,
            seed=self.config.seed,
        ).view(self.config.n_init, dataset.problem.dim)
        init_Y = dataset.problem(X=init_X).view(self.config.n_init, 1)
        return init_X, init_Y

    def fit_model(
        self,
        dataset: BODataset,
    ) -> Model:
        train_X, train_Y = dataset.get()[0]
        model = ModelFactory.create(
            **self.config.model,
            train_X=train_X,
            train_Y=train_Y,
        )
        mll = MLLFactory.create(**self.config.mll, model=model)
        fit_gpytorch_mll(mll)
        return model

    def get_next_data(
        self,
        dataset: BODataset,
        model: Model,
    ) -> tuple[Tensor, ...]:
        train_X, train_Y = dataset.get()[0]
        acqf = AcquisitionFactory.create(
            **self.config.acquisition,
            model=model,
            best_f=train_Y.max(),
        )
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor(dataset.problem._bounds, dtype=torch.double),
            q=self.config.q,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
        )
        new_X = candidates.detach()
        new_Y = dataset.problem(X=new_X)
        return new_X, new_Y
        
