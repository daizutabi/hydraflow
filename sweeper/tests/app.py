from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class A:
    x: float = 0


@dataclass
class Config:
    a: A = field(default_factory=A)
    b: int = 0
    c: str = "abc"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config) -> None:
    pass


if __name__ == "__main__":
    from hydra_plugins.hydra_ext_sweeper import override

    override(Config)
    app()
