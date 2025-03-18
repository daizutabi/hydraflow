from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def main():
    file = Path(sys.argv[-1])
    for line in file.read_text().splitlines():
        args = shlex.split(line)
        subprocess.run(["python", "app.py", *args], check=False)


if __name__ == "__main__":
    main()
