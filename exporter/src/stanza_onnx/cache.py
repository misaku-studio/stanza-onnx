"""Verified atomic downloads in Stanza's standard model-directory layout."""

import hashlib
from pathlib import Path
import os
import tempfile
from urllib.request import Request, urlopen

from . import RESOURCE_VERSION, __version__
from .catalog import resources, selected_sources


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_resources(cache_dir):
    # The reference pipeline is offline and must see the exact pinned catalog.
    from importlib.resources import files

    path = Path(cache_dir) / "resources.json"
    data = (
        files("stanza_onnx")
        .joinpath(f"data/resources_{RESOURCE_VERSION}.json")
        .read_bytes()
    )
    if not path.exists() or path.read_bytes() != data:
        atomic_write(path, data)


def checkpoint_path(cache_dir, language, source):
    return Path(cache_dir) / language / source["processor"] / f"{source['package']}.pt"


def verify_cache(language, cache_dir):
    paths = []
    for source in selected_sources(language):
        path = checkpoint_path(cache_dir, language, source)
        if not path.is_file():
            raise ValueError(
                f"Missing checkpoint: {path}; run 'stanza-onnx download' first"
            )
        if digest(path, "md5") != source["md5"]:
            raise ValueError(
                f"Checkpoint checksum mismatch: {path}; run 'stanza-onnx download' to repair it"
            )
        paths.append(path)
    return paths


def download_checkpoint(url, destination, expected_md5):
    """A failed transfer or checksum check never replaces an existing file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary)
    try:
        request = Request(url, headers={"User-Agent": f"stanza-onnx/{__version__}"})
        with (
            os.fdopen(descriptor, "wb") as output,
            urlopen(request, timeout=120) as response,
        ):
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if digest(temporary, "md5") != expected_md5:
            raise ValueError(f"Downloaded checkpoint checksum mismatch: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download(language, cache_dir):
    sources = selected_sources(language)
    write_resources(cache_dir)
    for source in sources:
        path = checkpoint_path(cache_dir, language, source)
        if not path.is_file() or digest(path, "md5") != source["md5"]:
            download_checkpoint(source["url"], path, source["md5"])
    return {
        "language": language,
        "cacheDir": str(Path(cache_dir).resolve()),
        "files": len(sources),
    }
