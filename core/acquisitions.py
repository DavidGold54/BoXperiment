# ============================================================================
# Acquisitions
# ============================================================================
import inspect

from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition import (
    analytic, monte_carlo, knowledge_gradient, max_value_entropy_search,
    predictive_entropy_search, 
)
import botorch.acquisition.multi_objective as mo


# ----------------------------------------------------------------------------
# Acquisition Factory
# ----------------------------------------------------------------------------
class AcquisitionFactory:
    """A factory that generates AcquisitionFunction objects from name strings.
    """

    REGISTRY: dict[str, AcquisitionFunction] = {
        "LogExpectedImprovement": analytic.LogExpectedImprovement,
        "LogProbabilityOfImprovement": analytic.LogProbabilityOfImprovement,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(acqf_cls: AcquisitionFunction):
            cls.REGISTRY[name] = acqf_cls
            return acqf_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        model: Model,
        **kwargs,
    ) -> AcquisitionFunction:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported acquisition type: {name}. ",
                f"Available acquisitions: {list(cls.REGISTRY.keys())}"
            )
        acqf_cls = cls.REGISTRY[name]
        sig = inspect.signature(acqf_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {"model": model}
        for key, value in kwargs.items():
            if key in args_list:
                params[key] = value
        return acqf_cls(**params)


# ----------------------------------------------------------------------------
# Custom Acquisitions
# ----------------------------------------------------------------------------
