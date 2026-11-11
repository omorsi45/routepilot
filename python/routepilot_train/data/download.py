"""Download and integrity check of the LLMRouterBench release tarball.

The file is about 1.3 GB, never committed, and kept under the gitignored data/raw.
"""

import argparse
import hashlib
import os
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from routepilot_train.data import selection

_CHUNK = 1 << 20
_TIMEOUT_SECONDS = 60


def verify(
    path: Path,
    *,
    expected_bytes: int = selection.DATASET_BYTES,
    expected_sha256: str = selection.DATASET_SHA256,
) -> str:
    """Check the size and SHA-256 of a file and return the hex digest.

    Raises ValueError when either differs. The size is compared first, so a
    truncated download fails without hashing it.
    """
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: size {size} bytes, expected {expected_bytes}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{path}: sha256 {actual}, expected {expected_sha256}")
    return actual


def fetch(
    dest: Path,
    *,
    url: str = selection.DATASET_URL,
    expected_bytes: int = selection.DATASET_BYTES,
    expected_sha256: str = selection.DATASET_SHA256,
) -> Path:
    """Make sure dest holds the verified dataset and return it.

    An existing dest that verifies is left alone with no network access. One that
    fails verification raises ValueError and is not touched. Otherwise the file is
    downloaded to dest + ".part" (resuming a previous partial download with a Range
    header), verified, and only then renamed into place. A part file that fails
    verification is deleted so the next call starts clean.
    """
    dest = Path(dest)
    if dest.exists():
        verify(dest, expected_bytes=expected_bytes, expected_sha256=expected_sha256)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_bytes:
        part.unlink()
        offset = 0
    if offset < expected_bytes:
        _download(url, part, offset)
    try:
        verify(part, expected_bytes=expected_bytes, expected_sha256=expected_sha256)
    except ValueError:
        part.unlink()
        raise
    os.replace(part, dest)
    return dest


def _download(url: str, part: Path, offset: int) -> None:
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        if offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"{url}: unexpected Content-Range {content_range!r}")
            mode = "ab"
        else:
            # A plain 200 means the server ignored the Range header: start over.
            mode = "wb"
        with part.open(mode) as f:
            shutil.copyfileobj(response, f, _CHUNK)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/bench-release.tar.gz"))
    args = parser.parse_args(argv)
    path = fetch(args.dest)
    print(f"{path}: {selection.DATASET_BYTES} bytes, sha256 {selection.DATASET_SHA256} (verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
