#!/usr/bin/env python3
"""Create one reproducible, scene-stratified episode-index manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from evaluation_io import sha256_file


def source_scene_id(scene_id: str) -> str:
    """Collapse ScanNet variants while preserving Matterport scene IDs."""
    value = str(scene_id).strip().split("/")[-1].split("\\")[-1]
    match = re.match(r"^(scene\d+)(?:[_-]\d+)?$", value)
    return match.group(1) if match else value


def make_manifest(
    episode_meta: List[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
    episode_meta_sha256: str,
    allowed_indices: Any = None,
    allowed_source: str = "",
) -> Dict[str, Any]:
    """Uniformly sample each source scene without replacement.

    ``allowed_indices`` restricts the candidate pool to a cohort (for example
    an adjusted-start manifest's eligible set) while keeping ``dataset_index``
    in the original ``episode_meta`` numbering, so the emitted manifest stays
    interchangeable with the unrestricted one.
    """
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    allowed = None if allowed_indices is None else {int(v) for v in allowed_indices}
    grouped: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for dataset_index, row in enumerate(episode_meta):
        if "scene_id" not in row:
            raise KeyError(f"episode_meta row {dataset_index} is missing scene_id")
        if allowed is not None and dataset_index not in allowed:
            continue
        exact_scene = str(row["scene_id"])
        grouped[source_scene_id(exact_scene)][exact_scene].append(dataset_index)
    if not grouped:
        raise ValueError("episode_meta is empty")

    selected: List[int] = []
    scene_stats: Dict[str, Dict[str, Any]] = {}
    for scene_id in sorted(grouped):
        exact_groups = grouped[scene_id]
        source_count = sum(len(indices) for indices in exact_groups.values())
        # Requested round-half-up rule, avoiding Python's bankers' round.
        count = int(math.floor(float(fraction) * source_count + 0.5))
        count = min(source_count, max(0, count))

        # Hamilton largest-remainder allocation preserves exact ScanNet variants.
        quotas: Dict[str, int] = {}
        remainders: List[Tuple[float, str]] = []
        for exact_scene in sorted(exact_groups):
            ideal = count * len(exact_groups[exact_scene]) / source_count
            quotas[exact_scene] = int(math.floor(ideal))
            remainders.append((ideal - quotas[exact_scene], exact_scene))
        remaining = count - sum(quotas.values())
        for _, exact_scene in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
            quotas[exact_scene] += 1

        exact_stats: Dict[str, Dict[str, int]] = {}
        for exact_scene in sorted(exact_groups):
            candidates = exact_groups[exact_scene]
            ranked = sorted(
                candidates,
                key=lambda dataset_index: (
                    hashlib.sha256(
                        f"{int(seed)}:{scene_id}:{exact_scene}:{dataset_index}".encode("utf-8")
                    ).digest(),
                    dataset_index,
                ),
            )
            chosen = ranked[: quotas[exact_scene]]
            selected.extend(chosen)
            exact_stats[exact_scene] = {
                "source_count": int(len(candidates)),
                "selected_count": int(len(chosen)),
            }
        scene_stats[scene_id] = {
            "source_count": int(source_count),
            "selected_count": int(count),
            "exact_scenes": exact_stats,
        }

    return {
        "manifest_schema_version": 1,
        "dataset_indices": sorted(selected),
        "source_scenes": scene_stats,
        "fraction": float(fraction),
        "seed": int(seed),
        "episode_meta_sha256": str(episode_meta_sha256),
        "source_count": int(sum(len(g) for scene in grouped.values() for g in scene.values())),
        "episode_meta_count": int(len(episode_meta)),
        "restricted_to": str(allowed_source),
        "selected_count": int(len(selected)),
        "sampling": "sha256_rank_without_replacement_per_exact_scene",
        "exact_scene_quota": "hamilton_largest_remainder_within_source_scene",
        "rounding": "floor(fraction*N+0.5)",
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {output_path}; pass --overwrite"
        )

    temporary_path = None
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
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Manifest appeared while writing: {output_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a fixed fraction independently from each v13 source scene"
    )
    parser.add_argument("--episode-meta", required=True, help="Path to episode_meta.json")
    parser.add_argument("--output", required=True, help="Output JSON manifest")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--restrict-to-manifest",
        dest="restrict_to_manifest",
        default=None,
        help="adjusted_start_manifest.npz; sample only its eligible dataset_index set",
    )
    args = parser.parse_args()

    episode_meta_path = Path(args.episode_meta).expanduser().resolve()
    with episode_meta_path.open("r", encoding="utf-8") as handle:
        episode_meta = json.load(handle)
    if not isinstance(episode_meta, list):
        raise ValueError("episode_meta.json must contain a list")
    allowed_indices = None
    allowed_source = ""
    if args.restrict_to_manifest:
        import numpy as np

        restrict_path = Path(args.restrict_to_manifest).expanduser().resolve()
        payload = np.load(restrict_path, allow_pickle=True)
        eligible = np.asarray(payload["eligible"], dtype=bool)
        allowed_indices = np.asarray(payload["dataset_index"])[eligible].tolist()
        allowed_source = str(restrict_path)

    manifest = make_manifest(
        episode_meta,
        fraction=args.fraction,
        seed=args.seed,
        episode_meta_sha256=sha256_file(episode_meta_path),
        allowed_indices=allowed_indices,
        allowed_source=allowed_source,
    )
    atomic_write_json(Path(args.output), manifest, overwrite=bool(args.overwrite))
    print(
        f"Saved {manifest['selected_count']}/{manifest['source_count']} episode indices "
        f"across {len(manifest['source_scenes'])} source scenes to {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
