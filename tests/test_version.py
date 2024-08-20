from importlib.metadata import version


def test_version():
    assert version("hydraflow").count(".") == 2
