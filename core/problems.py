# ============================================================================
# Problems
# ============================================================================
import inspect
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem
from botorch.exceptions.errors import InputDataError, UnsupportedError
from botorch.test_functions import (
    synthetic, multi_fidelity, multi_objective,
    multi_objective_multi_fidelity, sensitivity_analysis,
)


# ----------------------------------------------------------------------------
# Problem Factory
# ----------------------------------------------------------------------------
class ProblemFactory:

    REGISTRY: dict[str, BaseTestProblem] = {
        # Synthetic test functions
        "Branin" : synthetic.Branin
    }

    @classmethod
    def register(cls, name: str):
        def decorator(problem_cls: BaseTestProblem):
            cls.REGISTRY[name] = problem_cls
            return problem_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> BaseTestProblem:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported problem type: {name}. ",
                f"Available problems: {list(cls.REGISTRY.keys())}"
            )
        problem_cls = cls.REGISTRY[name]
        sig = inspect.signature(problem_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {}
        for key, value in kwargs.items():
            if key in args_list:
                params[key] = value
        return problem_cls(**params)


# ----------------------------------------------------------------------------
# Custom Problems
# ----------------------------------------------------------------------------
