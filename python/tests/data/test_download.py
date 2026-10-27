import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from routepilot_train.data import download

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _Server:
    def __init__(self, payload: bytes, *, honor_range: bool = True) -> None:
        self.requests: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server naming)
                range_header = self.headers.get("Range")
                outer.requests.append(range_header)
                start = 0
                if honor_range and range_header:
                    start = int(range_header.removeprefix("bytes=").rstrip("-"))
                body = payload[start:]
                self.send_response(206 if start else 200)
                if start:
                    last = len(payload) - 1
                    self.send_header("Content-Range", f"bytes {start}-{last}/{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/bench-release.tar.gz"

    def __enter__(self) -> "_Server":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join()


@pytest.fixture
def server() -> Iterator[_Server]:
    with _Server(PAYLOAD) as s:
        yield s


def fetch(dest: Path, url: str) -> Path:
    return download.fetch(
        dest, url=url, expected_bytes=len(PAYLOAD), expected_sha256=PAYLOAD_SHA256
    )


def test_verify_rejects_wrong_size(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD[:-1])
    with pytest.raises(ValueError, match="size"):
        download.verify(path, expected_bytes=len(PAYLOAD), expected_sha256=PAYLOAD_SHA256)


def test_fetch_skips_download_when_file_verifies(tmp_path, monkeypatch):
    dest = tmp_path / "bench-release.tar.gz"
    dest.write_bytes(PAYLOAD)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("fetch must not touch the network for a verified file")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert fetch(dest, "http://127.0.0.1:1/never-called") == dest


def test_fetch_downloads_and_verifies(tmp_path, server):
    dest = tmp_path / "raw" / "bench-release.tar.gz"
    assert fetch(dest, server.url) == dest
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_resumes_a_partial_download(tmp_path, server):
    dest = tmp_path / "bench-release.tar.gz"
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD[:4000])
    fetch(dest, server.url)
    assert dest.read_bytes() == PAYLOAD
    assert server.requests == ["bytes=4000-"]


def test_fetch_restarts_when_server_ignores_range(tmp_path):
    dest = tmp_path / "bench-release.tar.gz"
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD[:4000])
    with _Server(PAYLOAD, honor_range=False) as server:
        fetch(dest, server.url)
    assert dest.read_bytes() == PAYLOAD


def test_fetch_rejects_corrupt_download_and_discards_it(tmp_path):
    dest = tmp_path / "bench-release.tar.gz"
    with _Server(PAYLOAD[:-1] + b"!") as server, pytest.raises(ValueError, match="sha256"):
        fetch(dest, server.url)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_leaves_a_corrupt_existing_file_alone(tmp_path):
    dest = tmp_path / "bench-release.tar.gz"
    dest.write_bytes(PAYLOAD[:-1] + b"!")
    with pytest.raises(ValueError, match="sha256"):
        fetch(dest, "http://127.0.0.1:1/never-called")
    assert dest.read_bytes() == PAYLOAD[:-1] + b"!"
