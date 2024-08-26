import platform
from pathlib import Path
from urllib.parse import urlparse


def uri_to_path(uri: str) -> Path:
    """
    Convert a URI to a path.

    This function parses the given URI and converts it to a local file system
    path. On Windows, if the path starts with a forward slash, it is removed
    to ensure the path is correctly formatted.

    Args:
        uri (str): The URI to convert.

    Returns:
        Path: The path corresponding to the URI.
    """
    path = urlparse(uri).path
    if platform.system() == "Windows" and path.startswith("/"):
        path = path[1:]

    return Path(path)
