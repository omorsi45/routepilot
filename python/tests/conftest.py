"""Shared pytest configuration.

Default tests are hermetic like the Go suite: no internet. Loopback is allowed
so a test may run a local HTTP server.
"""

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any

import pytest


def _is_loopback(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX paths and other non-IP addresses never leave the machine
    host = address[0]
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname other than localhost would need DNS


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not _is_loopback(address):
            raise RuntimeError(f"hermetic tests may not connect to {address!r}")
        real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_loopback(address):
            raise RuntimeError(f"hermetic tests may not connect to {address!r}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield
