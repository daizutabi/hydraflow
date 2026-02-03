from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def main():
    file = Path(sys.argv[-1])
    for line in file.read_text(encoding="utf-8").splitlines():
        args = shlex.split(line)
        subprocess.run([sys.executable, "app.py", *args], check=False)


if __name__ == "__main__":
    main()
