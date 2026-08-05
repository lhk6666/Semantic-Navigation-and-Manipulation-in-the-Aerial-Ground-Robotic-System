"""Shared, side-effect-free I/O helpers for the two VLM evaluators."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


TRAJECTORY_ARCHIVE_FILENAME = "trajectory_archive.npz"
TRAJECTORY_ARCHIVE_SCHEMA_VERSION = 1
TRAJECTORY_COORDINATE_CONVENTION = "normalized_image_xy_x_right_y_down"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_backend_options(
    *, backend: str, model: str, api_key_env: Optional[str]
) -> Dict[str, Optional[str]]:
    """Validate public backend options without reading or returning a secret."""
    resolved_backend = str(backend).strip().lower()
    if resolved_backend not in {"vertexai", "genai"}:
        raise ValueError("backend must be one of: vertexai, genai")
    resolved_model = str(model).strip()
    if not resolved_model:
        raise ValueError("model must be non-empty")

    resolved_key_env: Optional[str] = None
    if resolved_backend == "genai":
        resolved_key_env = str(api_key_env or "GEMINI_API_KEY").strip()
        if not _ENV_NAME_RE.fullmatch(resolved_key_env):
            raise ValueError(
                "api-key-env must be an environment variable name, not a key value"
            )
    return {
        "backend": resolved_backend,
        "model": resolved_model,
        "api_key_env_name": resolved_key_env,
    }


def require_api_key_from_env(env_name: str) -> str:
    """Read the configured key while keeping diagnostics free of its value."""
    if not _ENV_NAME_RE.fullmatch(str(env_name)):
        raise ValueError("Invalid API-key environment variable name")
    value = os.environ.get(str(env_name))
    if not value:
        raise RuntimeError(f"Missing Google AI Studio API key in environment variable {env_name}")
    return value


def redact_api_key(message: Any, env_name: Optional[str]) -> str:
    """Remove the configured key from arbitrary exception text."""
    text = str(message)
    if env_name:
        secret = os.environ.get(str(env_name))
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episode_indices_file(path: Path, *, dataset_size: int) -> Tuple[List[int], Dict[str, Any]]:
    """Load an authoritative JSON manifest/list or whitespace text index list."""
    index_path = Path(path).expanduser().resolve()
    raw = index_path.read_bytes()
    if not raw.strip():
        raise ValueError(f"Episode-index file is empty: {index_path}")

    payload: Any
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        text = raw.decode("utf-8")
        tokens: List[str] = []
        for line in text.splitlines():
            line = line.split("#", 1)[0]
            tokens.extend(re.split(r"[\s,]+", line.strip()))
        payload = [token for token in tokens if token]

    manifest: Optional[Mapping[str, Any]] = payload if isinstance(payload, Mapping) else None
    if manifest is not None:
        if "dataset_indices" in manifest:
            payload = manifest["dataset_indices"]
        elif "episode_indices" in manifest:
            payload = manifest["episode_indices"]
        else:
            raise ValueError(
                "JSON index manifest must contain dataset_indices or episode_indices"
            )
    if not isinstance(payload, list):
        raise ValueError("Episode-index file must contain a JSON list or text integers")

    indices: List[int] = []
    for position, value in enumerate(payload):
        if isinstance(value, bool):
            raise ValueError(f"Boolean index at position {position} is invalid")
        try:
            index = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid index at position {position}: {value!r}") from error
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Non-integral index at position {position}: {value!r}")
        if str(value).strip() != str(index) and not isinstance(value, (int, np.integer)):
            raise ValueError(f"Invalid integer spelling at position {position}: {value!r}")
        if index < 0 or index >= int(dataset_size):
            raise IndexError(
                f"dataset_index {index} is outside [0, {int(dataset_size) - 1}]"
            )
        indices.append(index)

    if not indices:
        raise ValueError("Episode-index selection is empty")
    if len(set(indices)) != len(indices):
        raise ValueError("Episode-index selection contains duplicates")

    metadata: Dict[str, Any] = {
        "episode_indices_file": str(index_path),
        "episode_indices_file_sha256": hashlib.sha256(raw).hexdigest(),
        "episode_indices_file_count": len(indices),
    }
    if manifest is not None:
        for key in ("fraction", "seed", "episode_meta_sha256", "source_scenes"):
            if key in manifest:
                metadata[f"episode_indices_manifest_{key}"] = manifest[key]
    return indices, metadata


def select_episode_indices(
    *,
    dataset_size: int,
    num_episodes: Optional[int],
    stride: Optional[int],
    episode_indices_file: Optional[str],
) -> Tuple[List[int], Dict[str, Any]]:
    """Resolve evaluator indices; an explicit file is authoritative before a count cap."""
    size = int(dataset_size)
    if size <= 0:
        raise ValueError("dataset_size must be positive")
    if num_episodes is not None and int(num_episodes) <= 0:
        raise ValueError("num_episodes must be positive")

    if episode_indices_file:
        if stride not in (None, 1):
            raise ValueError("--episode-indices-file and --stride are mutually exclusive")
        indices, metadata = load_episode_indices_file(
            Path(episode_indices_file), dataset_size=size
        )
        if num_episodes is not None:
            indices = indices[: int(num_episodes)]
        metadata.update(
            {
                "selection_mode": "episode_indices_file",
                "selection_count": len(indices),
                "num_episodes_cap": None if num_episodes is None else int(num_episodes),
                "stride": None,
            }
        )
        return indices, metadata

    resolved_stride = 1 if stride is None else int(stride)
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")
    limit = size if num_episodes is None else min(int(num_episodes), size)
    indices = list(range(0, limit, resolved_stride))
    return indices, {
        "selection_mode": "stride",
        "selection_count": len(indices),
        "num_episodes_cap": None if num_episodes is None else int(num_episodes),
        "num_episodes_limit": limit,
        "stride": resolved_stride,
        "episode_indices_file": None,
        "episode_indices_file_sha256": None,
        "episode_indices_file_count": None,
    }


def build_trajectory_archive_payload(
    *,
    results: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Optional[np.ndarray]],
    episode_meta: Sequence[Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """Build the canonical, pickle-free single-trajectory baseline archive."""
    if len(results) != len(trajectories):
        raise ValueError("results/trajectories length mismatch")
    if not results:
        raise ValueError("Cannot build an empty trajectory archive")

    dataset_indices: List[int] = []
    sample_ids: List[str] = []
    scene_ids: List[str] = []
    successes: List[bool] = []
    errors: List[str] = []
    starts: List[np.ndarray] = []
    complete_paths: List[np.ndarray] = []

    for result, trajectory in zip(results, trajectories):
        dataset_index = int(result["dataset_index"])
        if dataset_index < 0 or dataset_index >= len(episode_meta):
            raise IndexError(f"dataset_index out of range: {dataset_index}")
        meta = episode_meta[dataset_index]
        sample_id = str(result["sample_id"])
        scene_id = str(result["scene_id"])
        if sample_id != str(meta["sample_id"]) or scene_id != str(meta["scene_id"]):
            raise ValueError(f"Episode identity mismatch at dataset_index={dataset_index}")

        start = np.asarray(meta["start"], dtype=np.float32)
        if start.shape != (2,) or not np.all(np.isfinite(start)):
            raise ValueError(f"Invalid start at dataset_index={dataset_index}")

        error_message = str(result.get("error") or "")
        success = trajectory is not None and not error_message
        if trajectory is None:
            pred = np.empty((0, 2), dtype=np.float32)
        else:
            pred = np.asarray(trajectory, dtype=np.float32)
            if pred.ndim != 2 or pred.shape[1] != 2 or not np.all(np.isfinite(pred)):
                raise ValueError(
                    f"Invalid trajectory at dataset_index={dataset_index}: {pred.shape}"
                )

        if len(pred) and np.array_equal(pred[0], start):
            complete = pred
        else:
            complete = np.concatenate((start[None, :], pred), axis=0)

        dataset_indices.append(dataset_index)
        sample_ids.append(sample_id)
        scene_ids.append(scene_id)
        successes.append(success)
        errors.append(error_message)
        starts.append(start)
        complete_paths.append(complete)

    dataset_index_array = np.asarray(dataset_indices, dtype=np.int64)
    if len(np.unique(dataset_index_array)) != len(dataset_index_array):
        raise ValueError("dataset_index values must be unique")

    lengths = np.asarray([len(path) for path in complete_paths], dtype=np.int64)
    padded = np.full((len(complete_paths), int(lengths.max()), 2), np.nan, dtype=np.float32)
    for row, path in enumerate(complete_paths):
        padded[row, : len(path)] = path

    return {
        "archive_schema_version": np.asarray(
            TRAJECTORY_ARCHIVE_SCHEMA_VERSION, dtype=np.int64
        ),
        "trajectory_includes_start": np.asarray(True, dtype=np.bool_),
        "coordinate_convention": np.asarray(TRAJECTORY_COORDINATE_CONVENTION),
        "dataset_index": dataset_index_array,
        "sample_id": np.asarray(sample_ids, dtype=str),
        "scene_id": np.asarray(scene_ids, dtype=str),
        "success": np.asarray(successes, dtype=np.bool_),
        "error": np.asarray(errors, dtype=str),
        "start_xy_norm": np.stack(starts).astype(np.float32, copy=False),
        "trajectory_xy_norm": padded,
        "trajectory_length": lengths,
    }


def save_trajectory_archive(
    path: Path,
    *,
    results: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Optional[np.ndarray]],
    episode_meta: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically write a canonical trajectory archive."""
    archive_path = Path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_trajectory_archive_payload(
        results=results, trajectories=trajectories, episode_meta=episode_meta
    )
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{archive_path.name}.",
            suffix=".tmp",
            dir=archive_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, archive_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file in its destination directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_evaluation_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Optional[np.ndarray]],
) -> None:
    """Persist completed rows and their raw predicted paths after each episode."""
    if len(results) != len(trajectories):
        raise ValueError("checkpoint results/trajectories length mismatch")
    records = []
    for result, trajectory in zip(results, trajectories):
        records.append(
            {
                "result": dict(result),
                "trajectory_xy_norm": (
                    None
                    if trajectory is None
                    else np.asarray(trajectory, dtype=float).tolist()
                ),
            }
        )
    atomic_write_json(
        Path(path),
        {
            "checkpoint_schema_version": 1,
            "config": dict(config),
            "episode_records": records,
        },
    )


def load_evaluation_checkpoint(
    path: Path,
    *,
    expected_config: Mapping[str, Any],
    episode_meta: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Optional[np.ndarray]]]:
    """Load a checkpoint only if config and episode identities match exactly."""
    checkpoint_path = Path(path)
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("checkpoint_schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if payload.get("config") != dict(expected_config):
        raise ValueError(
            "Checkpoint config differs from this run; use the exact original arguments"
        )
    records = payload.get("episode_records")
    if not isinstance(records, list):
        raise ValueError("Checkpoint episode_records must be a list")

    allowed = set(int(index) for index in selected_indices)
    seen = set()
    results: List[Dict[str, Any]] = []
    trajectories: List[Optional[np.ndarray]] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("result"), Mapping):
            raise ValueError("Invalid checkpoint record")
        result = dict(record["result"])
        dataset_index = int(result.get("dataset_index", -1))
        if dataset_index not in allowed or dataset_index in seen:
            raise ValueError(f"Unexpected/duplicate checkpoint index: {dataset_index}")
        meta = episode_meta[dataset_index]
        if (
            str(result.get("sample_id")) != str(meta["sample_id"])
            or str(result.get("scene_id")) != str(meta["scene_id"])
        ):
            raise ValueError(f"Checkpoint identity mismatch at index {dataset_index}")
        raw_trajectory = record.get("trajectory_xy_norm")
        trajectory: Optional[np.ndarray]
        if raw_trajectory is None:
            trajectory = None
        else:
            trajectory = np.asarray(raw_trajectory, dtype=np.float32)
            if (
                trajectory.ndim != 2
                or trajectory.shape[1] != 2
                or not np.all(np.isfinite(trajectory))
            ):
                raise ValueError(f"Invalid checkpoint trajectory at index {dataset_index}")
        if not result.get("error") and trajectory is None:
            raise ValueError(f"Successful checkpoint row lacks trajectory at index {dataset_index}")
        seen.add(dataset_index)
        results.append(result)
        trajectories.append(trajectory)
    return results, trajectories


def remove_error_rows_for_retry(
    results: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Optional[np.ndarray]],
) -> Tuple[List[Dict[str, Any]], List[Optional[np.ndarray]], List[int]]:
    """Drop completed error rows so resume schedules them again exactly once."""
    if len(results) != len(trajectories):
        raise ValueError("results/trajectories length mismatch")
    kept_results: List[Dict[str, Any]] = []
    kept_trajectories: List[Optional[np.ndarray]] = []
    removed_indices: List[int] = []
    seen = set()
    for result, trajectory in zip(results, trajectories):
        dataset_index = int(result["dataset_index"])
        if dataset_index in seen:
            raise ValueError(f"Duplicate dataset_index: {dataset_index}")
        seen.add(dataset_index)
        if result.get("error"):
            removed_indices.append(dataset_index)
        else:
            kept_results.append(dict(result))
            kept_trajectories.append(trajectory)
    return kept_results, kept_trajectories, removed_indices


def apply_adjusted_start_manifest(
    episode_meta: Sequence[MutableMapping[str, Any]],
    manifest_path: Path,
    *,
    dataset_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Replace each episode's start with its adjusted-start-manifest pose.

    Both ``DPZarrDataset.get_episode`` and
    ``build_trajectory_archive_payload`` read ``episode_meta[i]["start"]``, so
    mutating it in place is sufficient to make the rollout and the emitted
    archive agree on one hash-locked ``float32`` start.  The returned
    provenance lets the caller record the original poses, keeping the archive
    interchangeable with the CoFL/DP adjusted-start archives.
    """
    import numpy as np

    path = Path(manifest_path).expanduser().resolve()
    payload = np.load(path, allow_pickle=True)
    manifest_index = np.asarray(payload["dataset_index"], dtype=np.int64)
    if manifest_index.shape[0] != len(episode_meta):
        raise ValueError(
            f"manifest covers {manifest_index.shape[0]} episodes but "
            f"episode_meta has {len(episode_meta)}"
        )
    if not np.array_equal(manifest_index, np.arange(len(episode_meta))):
        raise ValueError("manifest dataset_index is not identity-aligned")
    eligible = np.asarray(payload["eligible"], dtype=bool)
    adjusted = np.asarray(payload["adjusted_start_xy_norm"], dtype=np.float32)
    manifest_sample_id = np.asarray([str(v) for v in payload["sample_id"]])

    targets = (
        range(len(episode_meta)) if dataset_indices is None
        else [int(v) for v in dataset_indices]
    )
    originals: Dict[int, Any] = {}
    replaced = 0
    for index in targets:
        row = episode_meta[index]
        if str(row["sample_id"]) != manifest_sample_id[index]:
            raise ValueError(f"sample_id mismatch at dataset_index={index}")
        if not eligible[index]:
            raise ValueError(
                f"dataset_index={index} is not eligible in {path.name}"
            )
        start = np.asarray(adjusted[index], dtype=np.float32)
        if start.shape != (2,) or not np.all(np.isfinite(start)):
            raise ValueError(f"invalid adjusted start at dataset_index={index}")
        originals[index] = np.asarray(row["start"], dtype=np.float32).tolist()
        row["start"] = start.tolist()
        replaced += 1

    return {
        "adjusted_start_manifest_path": str(path),
        "adjusted_start_manifest_sha256": sha256_file(path),
        "adjusted_start_manifest_schema_version": int(payload["schema_version"])
        if "schema_version" in payload.files
        else 1,
        "adjusted_start_replaced_episodes": int(replaced),
        "adjusted_start_original_xy_norm": originals,
    }


def augment_archive_with_start_provenance(
    archive_path: Path,
    provenance: Mapping[str, Any],
) -> None:
    """Add original starts and manifest identity to a saved archive."""
    import numpy as np

    path = Path(archive_path)
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    originals = provenance["adjusted_start_original_xy_norm"]
    payload["original_start_xy_norm"] = np.asarray(
        [originals[int(i)] for i in payload["dataset_index"]], dtype=np.float32
    )
    for key in (
        "adjusted_start_manifest_path",
        "adjusted_start_manifest_sha256",
        "adjusted_start_manifest_schema_version",
    ):
        payload[key] = np.asarray(provenance[key])
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w+b") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
