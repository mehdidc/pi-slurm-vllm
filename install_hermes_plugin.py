#!/usr/bin/env python3
"""Install the Hermes lifecycle and model-provider entry points as symlinks."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def install_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        print(f"already installed: {destination}")
        return
    if destination.exists() or destination.is_symlink():
        raise SystemExit(
            f"Refusing to replace existing plugin path: {destination}\n"
            "Remove or rename it explicitly, then run this installer again."
        )
    destination.symlink_to(source.resolve(), target_is_directory=True)
    print(f"installed: {destination} -> {source.resolve()}")


def ensure_local_api_key(hermes_home: Path) -> None:
    """Give Hermes's OpenAI client a non-secret key for the unauthenticated proxy."""
    env_file = hermes_home / ".env"
    existing = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    if re.search(r"^\s*PI_VLLM_API_KEY\s*=", existing, re.MULTILINE):
        print(f"kept existing PI_VLLM_API_KEY in {env_file}")
        return
    separator = "" if not existing or existing.endswith("\n") else "\n"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(
        f"{existing}{separator}PI_VLLM_API_KEY=pi-vllm-local\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    print(f"configured local proxy credential in {env_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")),
        help="Hermes profile home (default: HERMES_HOME or ~/.hermes)",
    )
    args = parser.parse_args()
    plugins = args.hermes_home.expanduser().resolve() / "plugins"
    install_link(
        REPO_ROOT / "extensions" / "hermes-vllm",
        plugins / "pi-slurm-vllm",
    )
    install_link(
        REPO_ROOT / "extensions" / "hermes-vllm-provider",
        plugins / "model-providers" / "hpc-vllm",
    )
    ensure_local_api_key(args.hermes_home.expanduser().resolve())
    print("\nEnable the lifecycle plugin with:")
    print("  hermes plugins enable pi-slurm-vllm")
    print("Then select a model with `hermes model` or `--provider hpc-vllm -m MODEL`.")


if __name__ == "__main__":
    main()
