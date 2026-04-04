import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root, optionally overridden for cluster setups."""
    env_root = os.getenv("MIST_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def expand_path(path_like: str | os.PathLike | None) -> str | None:
    """Expand env vars and user markers while preserving remote dataset ids."""
    if path_like is None:
        return None

    raw_path = os.fspath(path_like)
    if "://" in raw_path:
        return raw_path
    return os.path.expandvars(os.path.expanduser(raw_path))


def sampling_params_dir() -> Path:
    env_dir = os.getenv("MIST_SAMPLING_PARAMS_DIR")
    if env_dir:
        return Path(expand_path(env_dir)).resolve()
    return repo_root() / "sampling_params"


def model_registry_path() -> Path:
    env_path = os.getenv("MIST_MODEL_REGISTRY")
    if env_path:
        return Path(expand_path(env_path)).resolve()
    return repo_root() / "model_paths.txt"
