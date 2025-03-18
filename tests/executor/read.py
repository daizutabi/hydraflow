from __future__ import annotations

import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    src = Path(sys.argv[2])
    text = src.read_text()
    path.write_text(text)


if __name__ == "__main__":
    main()
