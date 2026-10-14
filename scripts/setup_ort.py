"""Fetch pinned, verified native ONNX Runtime CPU binaries for a deployment target."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

VERSION = "1.23.2"
ASSETS = {
    "darwin-arm64": (
        "onnxruntime-osx-arm64-1.23.2.tgz",
        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
    ),
    "darwin-x64": (
        "onnxruntime-osx-x86_64-1.23.2.tgz",
        "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6",
    ),
    "linux-arm64": (
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
    ),
    "linux-x64": (
        "onnxruntime-linux-x64-1.23.2.tgz",
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
    ),
    "win32-arm64": (
        "onnxruntime-win-arm64-1.23.2.zip",
        "1cfe88b6435df3b5fb0e9f6bd7d6f5df1e887b6174de7f6e2a47bab956f3f168",
    ),
    "win32-x64": (
        "onnxruntime-win-x64-1.23.2.zip",
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
    ),
}


def host_target():
    system = {"Darwin": "darwin", "Windows": "win32", "Linux": "linux"}.get(
        platform.system()
    )
    architecture = {
        "arm64": "arm64",
        "aarch64": "arm64",
        "x86_64": "x64",
        "amd64": "x64",
    }.get(platform.machine().lower())
    if not system or not architecture:
        raise ValueError("Unsupported host; choose --platform explicitly")
    return f"{system}-{architecture}"


def _extract_archive(archive, destination):
    """Reject ZIP traversal; tar extraction uses Python's data-only filter."""
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as contents:
            for name in contents.namelist():
                resolved = (destination / name).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive member")
            contents.extractall(destination)
    else:
        with tarfile.open(archive) as contents:
            contents.extractall(destination, filter="data")


def _copy_runtime_files(extracted, staging):
    licenses = staging / "licenses"
    licenses.mkdir()
    for file in extracted.rglob("*"):
        if not file.is_file() or file.is_symlink():
            continue
        if file.name.endswith((".dll", ".dylib")) or ".so" in file.name:
            shutil.copyfile(file, staging / file.name)
        elif "license" in file.name.lower() or "notice" in file.name.lower():
            shutil.copyfile(file, licenses / file.name)


def _library_name(target):
    if target.startswith("darwin"):
        return f"libonnxruntime.{VERSION}.dylib"
    if target.startswith("linux"):
        return f"libonnxruntime.so.{VERSION}"
    return "onnxruntime.dll"


def setup(output, target):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an absent or empty output directory")
    asset, checksum = ASSETS[target]
    url = (
        f"https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/{asset}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="stanza-onnx-native-", dir=output.parent
    ) as temporary:
        temporary = Path(temporary)
        archive = temporary / asset
        request = urllib.request.Request(url, headers={"User-Agent": "stanza-onnx"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            archive.open("wb") as destination,
        ):
            shutil.copyfileobj(response, destination)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
            raise ValueError("ONNX Runtime archive checksum mismatch")
        extracted = temporary / "extracted"
        extracted.mkdir()
        _extract_archive(archive, extracted)
        staging = temporary / "runtime"
        staging.mkdir()
        _copy_runtime_files(extracted, staging)
        library = _library_name(target)
        if not (staging / library).is_file():
            raise ValueError(f"Archive missing expected shared library {library}")
        metadata = {
            "version": VERSION,
            "platform": target,
            "source": url,
            "archiveSha256": checksum,
            "library": library,
        }
        (staging / "runtime.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            output.rmdir()
        staging.replace(output)
    return {**metadata, "libraryPath": str(output / library), "directory": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--platform", choices=ASSETS)
    args = parser.parse_args()
    try:
        print(json.dumps(setup(args.output, args.platform or host_target()), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"setup-ort: {error}\n")
