import platform
from pathlib import Path
from urllib.parse import urlparse


def uri_to_path(uri: str) -> Path:
    path = urlparse(uri).path
    if platform.system() == "Windows" and path.startswith("/"):
        path = path[1:]

    return Path(path)
