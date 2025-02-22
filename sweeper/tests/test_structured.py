def test_override() -> None:
    from hydra_plugins.hydra_ext_sweeper import override

    from .app import Config

    argv = ["a.x=1:2:5", "b=1:0.25:2:m", "c=1:2"]
    override(Config, argv)
    assert argv[0] == "a.x=1,3,5"
    assert argv[1] == "b=1e-3,1.25e-3,1.5e-3,1.75e-3,2.0e-3"
    assert argv[2] == "c=1:2"
