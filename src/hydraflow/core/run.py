"""Run module for HydraFlow.

This module provides the Run class, which represents an MLflow
Run in HydraFlow. A Run contains three main components:

1. info: Information about the run, which includes the run directory,
   run ID, and job name.
2. cfg: Configuration loaded from the Hydra configuration file.
3. impl: Implementation instance created by the provided
   factory function.

The Run class allows accessing these components through
a unified interface, and provides methods for setting default
configuration values and filtering runs.

The implementation instance (impl) can be created using a factory function
that accepts either just the artifacts directory path, or both the
artifacts directory path and the configuration instance. This flexibility
allows implementation classes to be configuration-aware and adjust their
behavior based on the run's configuration.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import MISSING
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast, overload

import polars as pl
from omegaconf import DictConfig, OmegaConf

from .run_info import RunInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars import DataFrame, Expr
    from polars._typing import PolarsDataType

    from .run_collection import RunCollection

# pyright: reportUnknownVariableType=false


class Run[C, I = None]:
    """Represent an MLflow Run in HydraFlow.

    A Run contains information about the run, configuration, and
    implementation. The configuration type C and implementation
    type I are specified as type parameters.
    """

    info: RunInfo
    """Information about the run, such as run directory, run ID, and job name."""

    impl_factory: Callable[[Path], I] | Callable[[Path, C], I]
    """Factory function to create the implementation instance.

    This can be a callable that accepts either:

    - A single Path parameter: the artifacts directory
    - Both a Path and a config parameter: the artifacts directory and
      the configuration instance

    The implementation dynamically detects the signature and calls the
    factory with the appropriate arguments.
    """

    def __init__(
        self,
        run_dir: Path,
        impl_factory: Callable[[Path], I] | Callable[[Path, C], I] | None = None,
    ) -> None:
        self.info = RunInfo(run_dir)
        self.impl_factory = impl_factory or (lambda _: None)  # pyright: ignore[reportAttributeAccessIssue]

    def __repr__(self) -> str:
        """Return a string representation of the Run."""
        class_name = self.__class__.__name__
        if isinstance(self.impl_factory, type):
            impl_name = f"[{self.impl_factory.__name__}]"
        else:
            impl_name = ""

        return f"{class_name}{impl_name}({self.info.run_id!r})"

    @cached_property
    def cfg(self) -> C:
        """The configuration instance loaded from the Hydra configuration file."""
        config_file = self.info.run_dir / "artifacts/.hydra/config.yaml"
        if config_file.exists():
            return OmegaConf.load(config_file)  # pyright: ignore[reportReturnType]

        return OmegaConf.create()  # pyright: ignore[reportReturnType]

    @cached_property
    def impl(self) -> I:
        """The implementation instance created by the factory function.

        This property dynamically examines the signature of the impl_factory
        using the inspect module and calls it with the appropriate arguments:

        - If the factory accepts one parameter: called with just the artifacts
          directory
        - If the factory accepts two parameters: called with the artifacts
          directory and the configuration instance

        This allows implementation classes to be configuration-aware and
        utilize both the file system and configuration information.
        """
        artifacts_dir = self.info.run_dir / "artifacts"

        sig = inspect.signature(self.impl_factory)
        params = list(sig.parameters.values())

        if len(params) == 1:
            impl_factory = cast("Callable[[Path], I]", self.impl_factory)
            return impl_factory(artifacts_dir)

        impl_factory = cast("Callable[[Path, C], I]", self.impl_factory)
        return impl_factory(artifacts_dir, self.cfg)

    @overload
    @classmethod
    def load(  # type: ignore
        cls,
        run_dir: str | Path,
        impl_factory: Callable[[Path], I] | Callable[[Path, C], I] | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def load(
        cls,
        run_dir: Iterable[str | Path],
        impl_factory: Callable[[Path], I] | Callable[[Path, C], I] | None = None,
        *,
        n_jobs: int = 0,
    ) -> RunCollection[Self]: ...

    @classmethod
    def load(
        cls,
        run_dir: str | Path | Iterable[str | Path],
        impl_factory: Callable[[Path], I] | Callable[[Path, C], I] | None = None,
        *,
        n_jobs: int = 0,
    ) -> Self | RunCollection[Self]:
        """Load a Run from a run directory.

        Args:
            run_dir (str | Path | Iterable[str | Path]): The directory where the
                MLflow runs are stored, either as a string, a Path instance,
                or an iterable of them.
            impl_factory (Callable[[Path], I] | Callable[[Path, C], I] | None):
                A factory function that creates the implementation instance. It
                can accept either just the artifacts directory path, or both the
                path and the configuration instance. Defaults to None, in which
                case a function that returns None is used.
            n_jobs (int): The number of parallel jobs. If 0 (default), runs
                sequentially. If -1, uses all available CPU cores.

        Returns:
            Self | RunCollection[Self]: A single Run instance or a RunCollection
            of Run instances.

        """
        if isinstance(run_dir, str | Path):
            return cls(Path(run_dir), impl_factory)

        from .run_collection import RunCollection

        if n_jobs == 0:
            runs = (cls(Path(r), impl_factory) for r in run_dir)
            return RunCollection(runs, cls.get)

        from joblib import Parallel, delayed

        parallel = Parallel(backend="threading", n_jobs=n_jobs)
        runs = parallel(delayed(cls)(Path(r), impl_factory) for r in run_dir)
        return RunCollection(runs, cls.get)  # pyright: ignore[reportArgumentType]

    @overload
    def update(
        self,
        key: str,
        value: Any | Callable[[Self], Any],
        *,
        force: bool = False,
    ) -> None: ...

    @overload
    def update(
        self,
        key: tuple[str, ...],
        value: Iterable[Any] | Callable[[Self], Iterable[Any]],
        *,
        force: bool = False,
    ) -> None: ...

    def update(
        self,
        key: str | tuple[str, ...],
        value: Any | Callable[[Self], Any],
        *,
        force: bool = False,
    ) -> None:
        """Set default value(s) in the configuration if they don't already exist.

        This method adds a value or multiple values to the configuration,
        but only if the corresponding keys don't already have values.
        Existing values will not be modified.

        Args:
            key: Either a string representing a single configuration path
                (can use dot notation like "section.subsection.param"),
                or a tuple of strings to set multiple related configuration
                values at once.
            value: The value to set. This can be:
                - For string keys: Any value, or a callable that returns
                  a value
                - For tuple keys: An iterable with the same length as the
                  key tuple, or a callable that returns such an iterable
                - For callable values: The callable must accept a single argument
                  of type Run (self) and return the appropriate value type
            force: Whether to force the update even if the key already exists.

        Raises:
            TypeError: If a tuple key is provided but the value is
                not an iterable, or if the callable doesn't return
                an iterable.

        """
        cfg: DictConfig = self.cfg  # pyright: ignore[reportAssignmentType]

        if isinstance(key, str):
            key = key.replace("__", ".")

            if force or OmegaConf.select(cfg, key, default=MISSING) is MISSING:
                v = value(self) if callable(value) else value  # type: ignore
                OmegaConf.update(cfg, key, v, force_add=True)
            return

        it = (OmegaConf.select(cfg, k, default=MISSING) is not MISSING for k in key)
        if not force and all(it):
            return

        if callable(value):
            value = value(self)  # type: ignore

        if not isinstance(value, Iterable) or isinstance(value, str):
            msg = f"{value} is not an iterable"
            raise TypeError(msg)

        for k, v in zip(key, value, strict=True):
            k_ = k.replace("__", ".")
            if force or OmegaConf.select(cfg, k_, default=MISSING) is MISSING:
                OmegaConf.update(cfg, k_, v, force_add=True)

    def get(self, key: str, default: Any | Callable[[Self], Any] = MISSING) -> Any:
        """Get a value from the information or configuration.

        Args:
            key: The key to look for. Can use dot notation for
                nested keys in configuration. Special keys:
                - "cfg": Returns the configuration object
                - "impl": Returns the implementation object
                - "info": Returns the run information object

            default: Value to return if the key is not found.
                If a callable, it will be called with the Run instance
                and the value returned will be used as the default.
                If not provided, AttributeError will be raised.

        Returns:
            Any: The value associated with the key, or the
            default value if the key is not found and a default
            is provided.

        Raises:
            AttributeError: If the key is not found and
                no default is provided.

        Note:
            The search order for keys is:

            1. Configuration (`cfg`)
            2. Implementation (`impl`)
            3. Run information (`info`)
            4. Run object itself (`self`)

        """
        key = key.replace("__", ".")

        value = OmegaConf.select(self.cfg, key, default=MISSING)  # pyright: ignore[reportArgumentType]
        if value is not MISSING:
            return value

        for attr in [self.impl, self.info, self]:
            value = getattr(attr, key, MISSING)
            if value is not MISSING and not callable(value):
                return value

        if default is not MISSING:
            if callable(default):
                return default(self)

            return default

        msg = f"No such key: {key}"
        raise AttributeError(msg)

    def lit(
        self,
        key: str,
        default: Any | Callable[[Self], Any] = MISSING,
        *,
        dtype: PolarsDataType | None = None,
    ) -> Expr:
        """Create a Polars literal expression from a run key.

        Args:
            key (str): The key to look up in the run's configuration or info.
            default (Any | Callable[[Run], Any], optional): Default value to
                use if the key is missing. If a callable is provided, it will be
                called with the Run instance.
            dtype (PolarsDataType | None): Explicit data type for the literal
                expression.

        Returns:
            Expr: A Polars literal expression aliased to the provided key.

        Raises:
            AttributeError: If the key is not found and no default is provided.

        """
        value = self.get(key, default)
        return pl.lit(value, dtype).alias(key)

    def to_frame(
        self,
        function: Callable[[Self], DataFrame],
        *keys: str | tuple[str, Any | Callable[[Self], Any]],
    ) -> DataFrame:
        """Convert the Run to a DataFrame.

        Args:
            function (Callable[[Run], DataFrame]): A function that takes a Run
                instance and returns a DataFrame.
            keys (str | tuple[str, Any | Callable[[Run], Any]]): The keys to
                add to the DataFrame.

        Returns:
            DataFrame: A DataFrame representation of the Run.

        """
        return function(self).with_columns(
            self.lit(k) if isinstance(k, str) else self.lit(k[0], k[1]) for k in keys
        )

    def to_dict(self, flatten: bool = True) -> dict[str, Any]:
        """Convert the Run to a dictionary.

        Args:
            flatten (bool, optional): If True, flattens nested dictionaries.
                Defaults to True.

        Returns:
            dict[str, Any]: A dictionary representation of the Run's configuration.

        """
        cfg = OmegaConf.to_container(self.cfg)
        if not isinstance(cfg, dict):
            raise TypeError("Configuration must be a dictionary")

        standard_dict: dict[str, Any] = {str(k): v for k, v in cfg.items()}  # pyright: ignore[reportUnknownArgumentType]

        if flatten:
            return _flatten_dict(standard_dict)

        return standard_dict

    @contextmanager
    def chdir(self, relative_dir: str = "") -> Iterator[Path]:
        """Change the current working directory to the artifact directory.

        This context manager changes the current working directory
        to the artifact directory of the run.
        It ensures that the directory is changed back
        to the original directory after the context is exited.

        Args:
            relative_dir (str): The relative directory to the artifact
                directory. Defaults to an empty string.

        Yields:
            Path: The artifact directory of the run.

        """
        artifacts_dir = self.info.run_dir / "artifacts" / relative_dir
        current_dir = Path.cwd()

        try:
            os.chdir(artifacts_dir)
            yield artifacts_dir

        finally:
            os.chdir(current_dir)

    def path(self, relative_path: str = "") -> Path:
        """Return the path relative to the artifact directory.

        Args:
            relative_path (str): The relative path to the artifact directory.

        Returns:
            Path: The path relative to the artifact directory.

        """
        return self.info.run_dir / "artifacts" / relative_path

    def iterdir(self, relative_dir: str = "") -> Iterator[Path]:
        """Iterate over the artifact directories for the run.

        Args:
            relative_dir (str): The relative directory to iterate over.

        Yields:
            Path: The artifact directory for the run.

        """
        yield from self.path(relative_dir).iterdir()

    def glob(self, pattern: str, relative_dir: str = "") -> Iterator[Path]:
        """Glob the artifact directories for the run.

        Args:
            pattern (str): The pattern to glob.
            relative_dir (str): The relative directory to glob.

        Yields:
            Path: The existing artifact paths that match the pattern.

        """
        yield from self.path(relative_dir).glob(pattern)


def _flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []

    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key).items())  # pyright: ignore[reportUnknownArgumentType]
        else:
            items.append((key, v))

    return dict(items)
