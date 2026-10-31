"""Facts the repository states in more than one place must agree.

The package metadata names the license of the repository, and the commands the docs and the
example configs give for hashing a key must exist in the shell the README tells Windows users
to use (Git Bash, which has ``sha256sum`` and no ``shasum``).
"""

import tomllib

from routepilot_train.runinfo import repo_root

ROOT = repo_root()
HASH_TEXTS = ("README.md", "examples/config.demo.yaml", "examples/config.local.yaml")


def test_the_package_license_is_the_license_of_the_repository() -> None:
    header = (ROOT / "LICENSE").read_text(encoding="utf-8")[:300]
    assert "Apache License" in header and "Version 2.0" in header
    meta = tomllib.loads((ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["license"] == "Apache-2.0"


def test_the_documented_key_hash_command_exists_in_git_bash() -> None:
    for name in HASH_TEXTS:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "printf %s your-key | sha256sum" in text, name
        assert "shasum -a 256" in text, f"{name} should name the macOS alternative"
