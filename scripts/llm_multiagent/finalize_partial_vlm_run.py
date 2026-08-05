#!/usr/bin/env python3
"""Write a VLM baseline's outputs from a partial atomic checkpoint.

The evaluators only emit ``results.json`` / ``metrics.csv`` /
``trajectory_archive.npz`` after the whole selected index set completes.  When a
run is stopped early — here, because the Gemini prepayment credits ran out — the
episodes that *did* finish are already durably recorded in
``evaluation_checkpoint.json``.  This closes such a run out over exactly those
episodes, using the same writers the evaluator uses, and records the partial
coverage explicitly so no downstream table can mistake it for a full run.

Resuming with a trimmed ``--episode-indices-file`` is not an option:
``load_evaluation_checkpoint`` requires the run config to compare equal, and the
selection config carries the index-file identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from evaluation_io import (
    TRAJECTORY_ARCHIVE_FILENAME,
    atomic_write_json,
    augment_archive_with_start_provenance,
    save_trajectory_archive,
)

# The per-scene reducer lives in each evaluator, not in evaluation_io. Both
# copies are identical; import the trajectory one and use it for either run.
from evaluate_vlm_trajectory import compute_per_scene_summary


def _mean(xs: List[float]) -> Any:
    return float(np.mean(xs)) if xs else None


def _std(xs: List[float]) -> Any:
    return float(np.std(xs)) if xs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--adjusted-start-manifest",
        type=Path,
        default=None,
        help="Recorded in the archive as start provenance, matching the run",
    )
    parser.add_argument("--selected-total", type=int, required=True,
                        help="Size of the originally selected index set")
    parser.add_argument("--stop-reason", type=str, default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint = json.loads((output_dir / "evaluation_checkpoint.json").read_text())
    if checkpoint.get("checkpoint_schema_version") != 1:
        raise SystemExit("Unsupported checkpoint schema")
    run_config = dict(checkpoint["config"])
    records = checkpoint["episode_records"]
    if isinstance(records, dict):
        records = list(records.values())

    episode_meta = json.loads((Path(args.dataset) / "episode_meta.json").read_text())

    # Each record holds the result row plus its trajectory; keep dataset order.
    rows: List[Dict[str, Any]] = []
    trajectories: List[Any] = []
    for record in records:
        result = dict(record.get("result", record))
        trajectory = record.get("trajectory_xy_norm", record.get("trajectory", record.get("pred_trajectory")))
        result.pop("trajectory", None)
        result.pop("pred_trajectory", None)
        rows.append(result)
        trajectories.append(
            None if trajectory is None else np.asarray(trajectory, dtype=np.float32)
        )
    order = np.argsort([int(r["dataset_index"]) for r in rows])
    rows = [rows[i] for i in order]
    trajectories = [trajectories[i] for i in order]

    start_provenance = None
    if args.adjusted_start_manifest:
        manifest = np.load(args.adjusted_start_manifest, allow_pickle=True)
        adjusted = np.asarray(manifest["adjusted_start_xy_norm"], dtype=np.float32)
        original = np.asarray(manifest["original_start_xy_norm"], dtype=np.float32)
        import hashlib

        digest = hashlib.sha256(
            Path(args.adjusted_start_manifest).read_bytes()
        ).hexdigest()
        # The rollout already used the adjusted pose, so make episode_meta agree
        # before the archive writer reads meta["start"].
        for row in rows:
            index = int(row["dataset_index"])
            episode_meta[index]["start"] = adjusted[index].tolist()
        start_provenance = {
            "adjusted_start_manifest_path": str(args.adjusted_start_manifest),
            "adjusted_start_manifest_sha256": digest,
            "adjusted_start_manifest_schema_version": int(manifest["schema_version"])
            if "schema_version" in manifest.files
            else 1,
            "adjusted_start_replaced_episodes": len(rows),
            "adjusted_start_original_xy_norm": {
                int(r["dataset_index"]): original[int(r["dataset_index"])].tolist()
                for r in rows
            },
        }

    valid = [r for r in rows if not r.get("error") and r.get("fge") != float("inf")]
    errors = sum(1 for r in rows if r.get("error"))
    fge = [float(r["fge"]) for r in valid]
    cr = [float(r["cr"]) for r in valid]
    plr = [float(r["plr"]) for r in valid]
    curv = [float(r["curv"]) for r in valid]

    coverage = {
        "partial_run": True,
        "completed_episodes": len(rows),
        "selected_episodes": int(args.selected_total),
        "coverage_fraction": len(rows) / float(args.selected_total),
        "stop_reason": args.stop_reason,
        "finalized_from": "evaluation_checkpoint.json",
    }

    atomic_write_json(
        output_dir / "results.json",
        {
            "config": run_config,
            "partial_coverage": coverage,
            "overall": {
                "fge_mean": _mean(fge), "fge_std": _std(fge),
                "cr_mean": _mean(cr),
                "plr_mean": _mean(plr), "plr_std": _std(plr),
                "curv_mean": _mean(curv), "curv_std": _std(curv),
                "num_valid": len(valid), "num_errors": errors,
                "num_with_mask": sum(1 for r in valid if r.get("has_mask")),
            },
            "episode_results": rows,
            "per_scene": compute_per_scene_summary(rows),
        },
    )

    archive_path = output_dir / TRAJECTORY_ARCHIVE_FILENAME
    save_trajectory_archive(
        archive_path, results=rows, trajectories=trajectories,
        episode_meta=episode_meta,
    )
    if start_provenance is not None:
        augment_archive_with_start_provenance(archive_path, start_provenance)

    with (output_dir / "metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "dataset_index,episode_idx,sample_id,scene_id,fge,cr,plr,curv,"
            "target_error,used_fallback,pred_traj_len,gt_traj_len,error\n"
        )
        for r in rows:
            handle.write(
                f"{r.get('dataset_index','')},{r.get('episode_idx','')},"
                f"{r.get('sample_id','')},{r.get('scene_id','')},{r.get('fge','')},"
                f"{r.get('cr','')},{r.get('plr','')},{r.get('curv','')},"
                f"{r.get('target_detection_error','')},{r.get('used_fallback','')},"
                f"{r.get('pred_traj_len','')},{r.get('gt_traj_len','')},"
                f"{str(r.get('error','')).replace(',', ' ')}\n"
            )

    atomic_write_json(output_dir / "partial_coverage.json", coverage)
    print(
        f"{output_dir.name}: finalized {len(rows)}/{args.selected_total} "
        f"({coverage['coverage_fraction']:.1%}), errors={errors}"
    )


if __name__ == "__main__":
    main()
