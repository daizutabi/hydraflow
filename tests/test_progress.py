from subprocess import run


def test_progress_bar():
    cp = run(["python", "-m", "hydraflow.progress"], capture_output=True, text=True)
    assert cp.returncode == 0
    # assert re.search(r"^\s+main.*100%.*$", cp.stdout, re.MULTILINE)
    # assert re.search(r"^.*unknown.*\d+/\?.*$", cp.stdout, re.MULTILINE)
    # assert re.search(r"^\s*#000.*100%.*$", cp.stdout, re.MULTILINE)
    # assert re.search(r"^unknown.*$", cp.stdout, re.MULTILINE)
    # assert "transient" not in cp.stdout
