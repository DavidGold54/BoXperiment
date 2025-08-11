# ============================================================================
# Models
# ============================================================================
import inspect

from torch import Tensor
from botorch.models.model import Model
from botorch.models import (
    cost, contextual, contextual_multioutput, fully_bayesian,
    fully_bayesian_multitask, gp_regression, gp_regression_mixed,
    higher_order_gp, latent_kronecker_gp, multitask, gp_regression_fidelity,
    pairwise_gp, relevance_pursuit, map_saas, robust_relevance_pursuit_model,
)

from core.model_utils import (MeanFactory, KernelFactory, LikelihoodFactory)


# ----------------------------------------------------------------------------
# Model Factory
# ----------------------------------------------------------------------------
class ModelFactory:
    """A factory that generates model objects from name strings.
    """

    REGISTRY: dict[str, Model] = {
        # GP regression models
        "SingleTaskGP": gp_regression.SingleTaskGP
    }

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Model):
            cls.REGISTRY[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        train_X: Tensor,
        train_Y: Tensor,
        **kwargs,
    ) -> Model:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported model type: {name}. ",
                f"Available models: {list(cls.REGISTRY.keys())}"
            )
        model_cls = cls.REGISTRY[name]
        sig = inspect.signature(model_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {"train_X": train_X, "train_Y": train_Y}
        for key, value in kwargs.items():
            if key in args_list:
                if key == "mean_module":
                    params[key] = MeanFactory.create(**value)
                elif key == "covar_module":
                    params[key] = KernelFactory.create(**value)
                elif key == "likelihood":
                    params[key] = LikelihoodFactory.create(**value)
                else:
                    params[key] = value
        return model_cls(**params)


# ----------------------------------------------------------------------------
# Custom Models
# ----------------------------------------------------------------------------