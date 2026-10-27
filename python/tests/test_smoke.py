import socket

import pytest

import routepilot_train


def test_version() -> None:
    assert routepilot_train.__version__ == "0.1.0"


def test_non_loopback_connections_are_blocked() -> None:
    # 192.0.2.1 is TEST-NET-1 (RFC 5737): never routable, and the conftest guard
    # must refuse it before any packet leaves the machine.
    with pytest.raises(RuntimeError, match="hermetic"):
        socket.create_connection(("192.0.2.1", 80), timeout=1)
