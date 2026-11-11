"""Facts the repository states in more than one place must agree.

The package metadata names the license of the repository, and the commands the docs and the
example configs give for hashing a key must exist in the shell the README tells Windows users
to use (Git Bash, which has ``sha256sum`` and no ``shasum``).
"""

import re
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


# --- pinned images -------------------------------------------------------------------------

IMAGE_LINE = re.compile(r"^\s*(?:image:\s*|FROM\s+)(?P<ref>\S+)", re.MULTILINE)
# Version components an image tag must carry, so a patch release cannot change a run unseen.
PINNED_COMPONENTS = {"redis": 3, "postgres": 2}


def image_refs(name: str) -> list[str]:
    text = (ROOT / name).read_text(encoding="utf-8")
    return [m.group("ref") for m in IMAGE_LINE.finditer(text)]


def test_service_images_name_an_exact_release() -> None:
    seen = set()
    for name in ("deploy/docker-compose.yml", ".github/workflows/integration.yml"):
        for ref in image_refs(name):
            repo, _, tag = ref.partition(":")
            if repo in PINNED_COMPONENTS:
                seen.add(repo)
                version = re.match(r"v?(\d+(?:\.\d+)*)", tag)
                assert version, ref
                assert len(version.group(1).split(".")) >= PINNED_COMPONENTS[repo], (name, ref)
    assert seen == set(PINNED_COMPONENTS)


def test_every_image_in_the_compose_stack_has_a_versioned_tag() -> None:
    for ref in image_refs("deploy/docker-compose.yml"):
        if ref.startswith("routepilot-compose:"):  # built from the Dockerfile, not pulled
            continue
        tag = ref.partition(":")[2]
        assert re.match(r"v?\d+\.\d+", tag), f"{ref} floats with its upstream"


def test_the_runtime_base_image_is_pinned_by_digest() -> None:
    final = image_refs("deploy/Dockerfile")[-1]
    assert re.search(r"@sha256:[0-9a-f]{64}$", final), final


def test_the_docs_do_not_claim_a_redis_password() -> None:
    compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "redis://redis:6379/0" in compose  # the compose Redis has no password
    for name in ("README.md", "docs/limitations.md", "deploy/docker-compose.yml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "Redis and PostgreSQL passwords" not in text, name
        assert "throwaway local passwords" not in text, name
