# ============================================================================
# Model Utilities
# ============================================================================
import inspect

from botorch.models.model import Model
from gpytorch import (
    constraints, priors, kernels, likelihoods, means, mlls
)

# ----------------------------------------------------------------------------
# Constraint Factory
# ----------------------------------------------------------------------------
class ConstraintFactory:
    """A factory that generates gpytorch constraint objects from name strings.
    """

    REGISTRY: dict[str, constraints.Interval] = {
        # Parameter constraints
        "Interval": constraints.Interval,
        "Positive": constraints.Positive,
        "LessThan": constraints.LessThan,
        "GreaterThan": constraints.GreaterThan,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(constraint_cls: constraints.Interval):
            cls.REGISTRY[name] = constraint_cls
            return constraint_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs,
    ) -> constraints.Interval:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported constraint type: {name}. ",
                f"Available constraints: {list(cls.REGISTRY.keys())}"
            )
        constraint_cls = cls.REGISTRY[name]
        sig = inspect.signature(constraint_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {}
        for key, value in kwargs.items():
            if key in args_list:
                params[key] = value
        return constraint_cls(**params)


# ----------------------------------------------------------------------------
# Prior Factory
# ----------------------------------------------------------------------------
class PriorFactory:
    """A factory that generates gpytorch prior objects from name strings.
    """

    REGISTRY: dict[str, priors.Prior] = {
        # Standard priors
        "GammaPrior": priors.GammaPrior,
        "HalfCauchyPrior": priors.HalfCauchyPrior,
        "LJKCovariancePrior": priors.LKJCovariancePrior,
        "MultivariateNormalPrior": priors.MultivariateNormalPrior,
        "NormalPrior": priors.NormalPrior,
        "SmoothedBoxPrior": priors.SmoothedBoxPrior,
        "UniformPrior": priors.UniformPrior,
        # Others
        "LogNormalPrior": priors.LogNormalPrior,
        "HalfNormalPrior": priors.HalfNormalPrior,
        "HorseshoePrior": priors.HorseshoePrior,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(prior_cls: priors.Prior):
            cls.REGISTRY[name] = prior_cls
            return prior_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs,
    ) -> priors.Prior:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported prior type: {name}. ",
                f"Available priors: {list(cls.REGISTRY.keys())}"
            )
        prior_cls = cls.REGISTRY[name]
        sig = inspect.signature(prior_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {}
        for key, value in kwargs.items():
            if key in args_list:
                params[key] = value
        return prior_cls(**params)


# ----------------------------------------------------------------------------
# Mean Factory
# ----------------------------------------------------------------------------
class MeanFactory:
    """A factory that generates gpytorch mean objects from name strings.
    """

    REGISTRY: dict[str, means.Mean] = {
        # Standard means
        "ZeroMean": means.ZeroMean,
        "ConstantMean": means.ConstantMean,
        "LinearMean": means.LinearMean,
        # Speciality means
        "MultitaskMean": means.MultitaskMean,
        "ConstantMeanGrad": means.ConstantMeanGrad,
        "ConstantMeanGradGrad": means.ConstantMeanGradGrad,
        "LinearMeanGrad": means.LinearMeanGrad,
        "LinearMeanGradGrad": means.LinearMeanGradGrad,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(mean_cls: means.Mean):
            cls.REGISTRY[name] = mean_cls
            return mean_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs,
    ) -> means.Mean:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported mean type: {name}. ",
                f"Available means: {list(cls.REGISTRY.keys())}"
            )
        mean_cls = cls.REGISTRY[name]
        sig = inspect.signature(mean_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {}
        for key, value in kwargs.items():
            if key in args_list:
                if key.endswith("constraint"):
                    params[key] = ConstraintFactory.create(**value)
                elif key.endswith("prior"):
                    params[key] = PriorFactory.create(**value)
                elif key.endswith("means"):
                    params[key] = [MeanFactory.create(**v) for v in value]
                else:
                    params[key] = value
        return mean_cls(**params)


# ----------------------------------------------------------------------------
# Kernel Factory
# ----------------------------------------------------------------------------
class KernelFactory:
    """A factory that generates gpytorch kernel objects from name strings.
    """

    REGISTRY: dict[str, kernels.Kernel] = {
        # Standard kernels
        "ConstantKernel": kernels.ConstantKernel,
        "CosineKernel": kernels.CosineKernel,
        "CylindricalKernel": kernels.CylindricalKernel,
        "LinearKernel": kernels.LinearKernel,
        "MaternKernel": kernels.MaternKernel,
        "PeriodicKernel": kernels.PeriodicKernel,
        "PiecewisePolynomialKernel": kernels.PiecewisePolynomialKernel,
        "PolynomialKernel": kernels.PolynomialKernel,
        "PolynomialKernelGrad": kernels.PolynomialKernelGrad,
        "RBFKernel": kernels.RBFKernel,
        "RQKernel": kernels.RQKernel,
        "SpectralDeltaKernel": kernels.SpectralDeltaKernel,
        "SpectralMixtureKernel": kernels.SpectralMixtureKernel,
        # Composition/Decoration kernels
        "AdditiveKernel": kernels.AdditiveKernel,
        "AdditiveStructureKernel": kernels.AdditiveStructureKernel,
        "ProductKernel": kernels.ProductKernel,
        "ScaleKernel": kernels.ScaleKernel,
        # Speciality kernels
        "ArcKernel": kernels.ArcKernel,
        "HammingMQKernel": kernels.HammingIMQKernel,
        "IndexKernel": kernels.IndexKernel,
        "LCMKernel": kernels.LCMKernel,
        "MultitaskKernel": kernels.MultitaskKernel,
        "RBFKernelGrad": kernels.RBFKernelGrad,
        "RBFKernelGradGrad": kernels.RBFKernelGradGrad,
        "Matern52KernelGrad": kernels.Matern52KernelGrad,
        # Kernels for scalable GP regression methods
        "GridKernel": kernels.GridKernel,
        "GridInterpolationKernel": kernels.GridInterpolationKernel,
        "InducingPointKernel": kernels.InducingPointKernel,
        "RFFKernel": kernels.RFFKernel,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(kernel_cls: kernels.Kernel):
            cls.REGISTRY[name] = kernel_cls
            return kernel_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> kernels.Kernel:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported kernel type: {name}. ",
                f"Available kernels: {list(cls.REGISTRY.keys())}"
            )
        kernel_cls = cls.REGISTRY[name]
        sig = inspect.signature(kernel_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {}
        for key, value in kwargs.items():
            if key in args_list:
                if key.endswith("constraint"):
                    params[key] = ConstraintFactory.create(**value)
                elif key.endswith("prior"):
                    params[key] = PriorFactory.create(**value)
                elif key.endswith("kernel") or key.endswith("covar_module"):
                    params[key] = KernelFactory.create(**value)
                elif key.endswith("kernels"):
                    params[key] = [KernelFactory.create(**v) for v in value]
                else:
                    params[key] = value
        return kernel_cls(**params)


# ----------------------------------------------------------------------------
# Likelihood Factory
# ----------------------------------------------------------------------------
class LikelihoodFactory:
    """A factory that generates gpytorch likelihood objects from name strings.
    """

    REGISTRY: dict[str, likelihoods.Likelihood] = {
        # One-Dimensional likelihoods
        "GaussianLikelihood": likelihoods.GaussianLikelihood,
        "FixedNoiseGaussianLikelihood": likelihoods.FixedNoiseGaussianLikelihood,
        "DirichletClassificationLikelihood": likelihoods.DirichletClassificationLikelihood,
        "BernoulliLikelihood": likelihoods.BernoulliLikelihood,
        "BetaLikelihood": likelihoods.BetaLikelihood,
        "LaplaceLikelihood": likelihoods.LaplaceLikelihood,
        "StudentTLikelihood": likelihoods.StudentTLikelihood,
        # Multi-Dimensional likelihoods
        "MultitaskGaussianLikelihood": likelihoods.MultitaskGaussianLikelihood,
        "SoftmaxLikelihood": likelihoods.SoftmaxLikelihood,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(likelihood_cls: likelihoods.Likelihood):
            cls.REGISTRY[name] = likelihood_cls
            return likelihood_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> likelihoods.Likelihood:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported likelihood type: {name}. ",
                f"Available likelihoods: {list(cls.REGISTRY.keys())}"
            )
        likelihood_cls = cls.REGISTRY[name]
        sig = inspect.signature(likelihood_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {}
        for key, value in kwargs.items():
            if key in args_list:
                if key.endswith("constraint"):
                    params[key] = ConstraintFactory.create(**value)
                elif key.endswith("prior"):
                    params[key] = PriorFactory.create(**value)
                else:
                    params[key] = value
        return likelihood_cls(**params)


# ----------------------------------------------------------------------------
# Marginal Log Likelihood (MLL) Factory
# ----------------------------------------------------------------------------
class MLLFactory:
    """A factory that generates gpytorch MLL objects from name strings.
    """

    REGISTRY: dict[str, mlls.MarginalLogLikelihood] = {
        # Exact GP Inference
        "ExactMarginalLogLikelihood": mlls.ExactMarginalLogLikelihood,
        "LeaveOneOutPseudoLikelihood": mlls.LeaveOneOutPseudoLikelihood,
        # Approximate GP Inference
        "VariationalELBO": mlls.VariationalELBO,
        "PredictiveLogLikelihood": mlls.PredictiveLogLikelihood,
        "GammaRobustVariationalELBO": mlls.GammaRobustVariationalELBO,
        "DeepApproximateMLL": mlls.DeepApproximateMLL,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(mll_cls: mlls.MarginalLogLikelihood):
            cls.REGISTRY[name] = mll_cls
            return mll_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        model: Model,
        **kwargs
    ) -> mlls.MarginalLogLikelihood:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported mll type: {name}. ",
                f"Available mlls: {list(cls.REGISTRY.keys())}"
            )
        mll_cls = cls.REGISTRY[name]
        sig = inspect.signature(mll_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {}
        for key, value in kwargs.items():
            if key in args_list:
                params[key] = value
        return mll_cls(model.likelihood, model, **params)