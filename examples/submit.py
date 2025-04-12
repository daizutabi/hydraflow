import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_file, opt_file = sys.argv[1:]
    text = Path(opt_file).read_text()

    for line in text.splitlines():
        opts = shlex.split(line)
        args = [sys.executable, app_file, *opts]
        print(args)
        subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
