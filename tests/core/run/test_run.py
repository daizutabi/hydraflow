import pytest
from omegaconf import DictConfig

from hydraflow.core.run import _set_default


class Db:
    name: str
    b: int


class Config:
    a: int
    db: Db


@pytest.fixture
def cfg():
    return DictConfig({})


def test_set_default(cfg: DictConfig):
    _set_default(None, cfg, "a", 10)
    assert cfg.a == 10
    _set_default(None, cfg, "db.name", "abc")
    assert cfg.db.name == "abc"
    _set_default(None, cfg, "a", 20)
    assert cfg.a == 10
    _set_default(None, cfg, "db.name", "def")
    assert cfg.db.name == "abc"
    _set_default(None, cfg, "db.b", 100)
    assert cfg.db.b == 100


def test_run_config_set_default_callable(cfg: DictConfig):
    _set_default(None, cfg, "db.name", "abc")
    assert cfg.db.name == "abc"
    _set_default(None, cfg, "db.b", lambda x: len(cfg.db.name))
    assert cfg.db.b == 3
    _set_default(None, cfg, "a", lambda x: cfg.db.b * 10)
    assert cfg.a == 30


def test_run_config_set_default_tuple(cfg: DictConfig):
    _set_default(None, cfg, ("db.name", "db.b"), ["xyz", 1000])
    assert cfg.db.name == "xyz"
    assert cfg.db.b == 1000
    _set_default(None, cfg, ("db.name", "db.b"), ["XYZ", 2000])
    assert cfg.db.name == "xyz"
    assert cfg.db.b == 1000


def test_run_config_set_default_tuple_callable(cfg: DictConfig):
    _set_default(None, cfg, ("db.name", "db.b"), lambda x: ["a", 1])
    assert cfg.db.name == "a"
    assert cfg.db.b == 1
    _set_default(None, cfg, ("db.name", "db.b"), lambda x: [1 / 0, 1 / 0])


def test_run_config_set_default_tuple_error(cfg: DictConfig):
    with pytest.raises(TypeError):
        _set_default(None, cfg, ("db.name", "db.b"), lambda x: "ab")


text = """\
    - cx=5e-09
    - cz=5e-09
  job:
    name: fine_0204
    chdir: null
    override_dirname: Bext=-0.008,cx=5e-09,cz=5e-09,width=3e-06
    id: '0'
    num: 0
"""


def test_job_name(tmp_path_factory: pytest.TempPathFactory):
    from hydraflow.core.run import get_job_name

    p = tmp_path_factory.mktemp("artifacts", numbered=False)
    (p / ".hydra").mkdir()
    (p / ".hydra/hydra.yaml").write_text(text)
    assert get_job_name(p.parent) == "fine_0204"
