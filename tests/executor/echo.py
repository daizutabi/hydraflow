from __future__ import annotations

import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    arg = " ".join(sys.argv[3:5])

    if not path.exists():
        path.write_text(arg)
    else:
        text = path.read_text()
        path.write_text(f"{text} {arg}")


if __name__ == "__main__":
    main()
