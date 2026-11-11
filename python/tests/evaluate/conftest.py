"""Fixtures for the tests that run the Go ``routepilot replay`` command."""

from pathlib import Path

import pytest

from routepilot_train import replaytool
from routepilot_train.runinfo import repo_root


@pytest.fixture(scope="session")
def binary() -> Path:
    """The routepilot binary built from this checkout.

    Skips when no Go toolchain that can build the module is installed; any other build
    failure fails the test.
    """
    try:
        return replaytool.build_go(repo_root())
    except replaytool.GoToolchainError as err:
        pytest.skip(str(err))
