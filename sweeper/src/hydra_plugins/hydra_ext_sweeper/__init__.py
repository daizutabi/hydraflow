import sys

from omegaconf import OmegaConf


def override(config: object, argv: list[str] | None = None) -> None:
    from ._parser import parse
    from .ext_sweeper import ExtSweeper

    if argv is None:
        argv = sys.argv

    cfg = OmegaConf.structured(config)

    parsed = []
    for arg in argv:
        if "=" not in arg:
            parsed.append(arg)
            continue

        key, value = arg.split("=")

        if ExtSweeper.is_number(cfg, key):
            value = parse(value)
            parsed.append(f"{key}={value}")
        else:
            parsed.append(arg)

    argv[:] = parsed
