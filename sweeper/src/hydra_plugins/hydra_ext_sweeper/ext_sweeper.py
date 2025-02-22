import logging
from collections.abc import Sequence
from dataclasses import dataclass

from hydra._internal.core_plugins.basic_sweeper import BasicSweeper, BasicSweeperConf
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import Override, OverrideType, ValueType
from hydra.core.utils import JobReturn
from hydra.errors import ConfigCompositionException
from hydra.types import RunMode
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@dataclass
class ExtSweeperConf(BasicSweeperConf):
    _target_: str = "hydra_plugins.hydra_ext_sweeper.ext_sweeper.ExtSweeper"


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="ext",
    node=ExtSweeperConf,
    provider="hydra-ext-sweeper",
)

log = logging.getLogger(__name__)


class ExtSweeper(BasicSweeper):
    """A hydra sweeper with extended syntax for efficient parameter sweeping."""

    def sweep(self, arguments: list[str]) -> list[Sequence[JobReturn]]:
        from ._parser import parse

        assert self.config is not None
        assert self.hydra_context is not None

        config_loader = self.hydra_context.config_loader

        try:
            config = config_loader.load_configuration(
                config_name=self.config.hydra.job.config_name,
                overrides=[],
                run_mode=RunMode.RUN,
            )
        except ConfigCompositionException:
            return super().sweep(arguments)

        parser = OverridesParser.create(config_loader)

        parsed = []
        for argument in arguments:
            override = parser.parse_override(argument)
            key = override.get_key_element()

            if self.is_extended(override) and self.is_number(config, key):
                value = parse(override.get_value_string())
                parsed.append(f"{key}={value}")
            else:
                parsed.append(argument)

        return super().sweep(parsed)

    @staticmethod
    def is_extended(override: Override) -> bool:
        is_change = override.type is OverrideType.CHANGE
        is_element = override.value_type is ValueType.ELEMENT
        return is_change and is_element

    @staticmethod
    def is_number(cfg: DictConfig, key: str) -> bool:
        for key_ in key.split("."):
            if not isinstance(cfg, DictConfig) or key_ not in cfg:
                return False

            cfg = cfg[key_]

        return isinstance(cfg, int | float)
