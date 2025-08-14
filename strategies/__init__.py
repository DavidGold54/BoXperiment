# ============================================================================
# BO Strategies
# ============================================================================
import inspect
from types import SimpleNamespace

from strategies.base import BaseBOStrategy, BasicBOStrategy


class BOStrategyFactory:

    REGISTRY: dict[str, BaseBOStrategy] = {
        "Basic": BasicBOStrategy,
    }

    @classmethod
    def register(cls, name: str):
        def decorator(strategy_cls: BaseBOStrategy):
            cls.REGISTRY[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        config: SimpleNamespace,
        **kwargs,
    ) -> BaseBOStrategy:
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unsupported strategy type: {name}. ",
                f"Available strategies: {list(cls.REGISTRY.keys())}"
            )
        strategy_cls = cls.REGISTRY[name]
        sig = inspect.signature(strategy_cls.__init__).parameters.values()
        args_list = [p.name for p in sig if p.name != "self"]
        params = {"config": config}
        for key, value in kwargs.items():
            if key in args_list:
                params[key] = value
        return strategy_cls(**params)

