from pathlib import Path


def run():
    Path("watch.txt").write_text("watch")


if __name__ == "__main__":
    run()
