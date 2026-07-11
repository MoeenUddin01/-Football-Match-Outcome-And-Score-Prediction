"""Utilities for persisting artifacts used across the prediction pipeline."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_artifact(obj: Any, name: str, artifacts_dir: str | Path) -> Path:
    """Persist an object as JSON when possible, otherwise as a pickle file.

    Parameters
    ----------
    obj:
        Python object to persist.
    name:
        Artifact name. The extension is inferred from the content type when
        no explicit suffix is supplied.
    artifacts_dir:
        Target directory for the artifact.

    Returns
    -------
    pathlib.Path
        Path to the saved artifact.
    """
    artifact_dir = Path(artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = Path(name)
    if artifact_name.suffix.lower() in {".json"}:
        target_path = artifact_dir / artifact_name.name
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, default=str)
    elif artifact_name.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
        target_path = artifact_dir / artifact_name.name
        with target_path.open("wb") as handle:
            pickle.dump(obj, handle)
    else:
        try:
            json.dumps(obj)
        except TypeError:
            target_path = artifact_dir / f"{artifact_name.name}.pkl"
            with target_path.open("wb") as handle:
                pickle.dump(obj, handle)
        else:
            target_path = artifact_dir / f"{artifact_name.name}.json"
            with target_path.open("w", encoding="utf-8") as handle:
                json.dump(obj, handle, indent=2, default=str)

    logger.info("Saved artifact to %s", target_path)
    return target_path


def load_artifact(name: str, artifacts_dir: str | Path) -> Any:
    """Load a previously saved artifact from JSON or pickle storage."""
    artifact_dir = Path(artifacts_dir)
    artifact_name = Path(name)

    if artifact_name.suffix.lower() in {".json"}:
        candidate_paths = [artifact_dir / artifact_name.name]
    elif artifact_name.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
        candidate_paths = [artifact_dir / artifact_name.name]
    else:
        candidate_paths = [
            artifact_dir / f"{artifact_name.name}.json",
            artifact_dir / f"{artifact_name.name}.pkl",
            artifact_dir / f"{artifact_name.name}.pickle",
            artifact_dir / f"{artifact_name.name}.joblib",
        ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            if candidate_path.suffix.lower() == ".json":
                with candidate_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            with candidate_path.open("rb") as handle:
                return pickle.load(handle)

    raise FileNotFoundError(f"Artifact '{name}' was not found in {artifact_dir}")
