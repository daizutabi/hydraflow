import os
import sys
import time
from pathlib import Path


def run():
    for k in range(2):
        print(k, time.time(), flush=True)
        Path(f"{k}.txt").write_text(f"{k} {time.time()}")
        time.sleep(1)

    for k in range(2, 4):
        print(k, time.time(), file=sys.stderr, flush=True)
        Path(f"{k}.txt").write_text(f"{k} {time.time()}")
        time.sleep(1)

    os.makedirs("a/b")
    Path("a/b/c.txt").write_text(f"4 {time.time()}")


if __name__ == "__main__":
    run()
