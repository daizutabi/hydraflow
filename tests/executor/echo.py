from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    arg = " ".join(sys.argv[3:5])

    if not path.exists():
        path.write_text(arg, encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
        path.write_text(f"{text} {arg}", encoding="utf-8")


if __name__ == "__main__":
    main()
