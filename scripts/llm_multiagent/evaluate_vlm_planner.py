#!/usr/bin/env python3
"""Minimal VLM+Planner evaluation (multi-threaded).

Pipeline per episode:
  1) Load semantic-map image + GT start/goal/trajectory from Zarr
  2) Call SemanticMapPlanner.analyze_scene(image, instruction)
  3) Plan with PathPlanner.generate_safe_path_astar using obstacle bboxes
  4) Compute metrics and save JSON/CSV

Intentionally minimal: no checkpointing, no visualization, no heavy retry/backoff.
"""

import argparse
import importlib.util
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import zarr
from tqdm import tqdm


# Add paths for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "planners"))

from path_planner import PathPlanner
from semantic_map_planner import SemanticMapPlanner


thread_local = threading.local()


def get_vlm_planner() -> SemanticMapPlanner:
    if not hasattr(thread_local, "vlm_planner"):
        thread_local.vlm_planner = SemanticMapPlanner()
    return thread_local.vlm_planner


def get_path_planner() -> PathPlanner:
    if not hasattr(thread_local, "path_planner"):
        thread_local.path_planner = PathPlanner()
    return thread_local.path_planner


def _load_flowvla_trajectory_metrics_module():
    """Load FlowVLA's shared trajectory_metrics.py to avoid metric drift."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "scripts" / "test" / "trajectory_metrics.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("flowvla_trajectory_metrics", str(candidate))
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec for {candidate}")
            module = importlib.util.module_from_spec(spec)
            # Python 3.12 dataclasses expects the module to be registered.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "Could not locate FlowVLA/scripts/test/trajectory_metrics.py from this script location."
    )


_FLOWVLA_METRICS = _load_flowvla_trajectory_metrics_module()
TrajectoryMetrics = _FLOWVLA_METRICS.TrajectoryMetrics
STANDARD_NUM_WAYPOINTS = _FLOWVLA_METRICS.STANDARD_NUM_WAYPOINTS


def scene_group_id(scene_id: str) -> str:
    """Group scene variants like scene700_01/scene700_02 into scene700.

    For non-matching scene IDs (e.g., random hashes), returns the original.
    """
    if not scene_id:
        return "unknown"
    s = str(scene_id).strip()
    s = s.split("/")[-1].split("\\")[-1]
    m = re.match(r"^(scene\d+)(?:[_-]\d+)?$", s)
    if m:
        return m.group(1)
    parts = re.split(r"[_-]", s)
    if parts and re.fullmatch(r"scene\d+", parts[0]):
        return parts[0]
    m = re.match(r"^(scene\d+)", s)
    if m:
        return m.group(1)
    return s


def compute_per_scene_summary(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute per-scene metric summary from episode-level results."""
    valid = [r for r in results if not r.get("error") and r.get("fge") != float("inf")]
    scene_results: Dict[str, List[Dict[str, Any]]] = {}
    for r in valid:
        sid = scene_group_id(r.get("scene_id", "unknown"))
        scene_results.setdefault(sid, []).append(r)

    def _mean(xs: List[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    def _std(xs: List[float]) -> Optional[float]:
        return float(np.std(xs)) if xs else None

    summary: List[Dict[str, Any]] = []
    for sid, scene_res in sorted(scene_results.items()):
        fge_vals = [float(x["fge"]) for x in scene_res]
        cr_vals = [float(x["cr"]) for x in scene_res]
        plr_vals = [float(x["plr"]) for x in scene_res]
        curv_vals = [float(x["curv"]) for x in scene_res]
        tgt_errs = [
            float(x["target_detection_error"])
            for x in scene_res
            if x.get("target_detection_error") is not None
        ]

        summary.append(
            {
                "scene_id": sid,
                "num_episodes": int(len(scene_res)),
                "fge": _mean(fge_vals),
                "fge_std": _std(fge_vals),
                "cr": _mean(cr_vals),
                "cr_std": _std(cr_vals),
                "plr": _mean(plr_vals),
                "plr_std": _std(plr_vals),
                "curv": _mean(curv_vals),
                "curv_std": _std(curv_vals),
                "target_error": _mean(tgt_errs),
                "target_error_std": _std(tgt_errs),
            }
        )
    return summary


def postprocess_existing_results(*, results_path: Path, output_dir: Path, inplace: bool) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    episode_results = blob.get("episode_results") or []
    if not isinstance(episode_results, list):
        raise ValueError("results.json missing 'episode_results' list")

    per_scene = compute_per_scene_summary(episode_results)
    blob["per_scene"] = per_scene

    scene_csv_file = output_dir / "scene_metrics.csv"
    with open(scene_csv_file, "w", encoding="utf-8") as f:
        f.write("scene_id,num_episodes,fge,fge_std,cr,cr_std,plr,plr_std,curv,curv_std,target_error,target_error_std\n")
        for s in per_scene:
            f.write(
                f"{s.get('scene_id','')},{s.get('num_episodes','')},"
                f"{s.get('fge','')},{s.get('fge_std','')},"
                f"{s.get('cr','')},{s.get('cr_std','')},"
                f"{s.get('plr','')},{s.get('plr_std','')},"
                f"{s.get('curv','')},{s.get('curv_std','')},"
                f"{s.get('target_error','')},{s.get('target_error_std','')}\n"
            )
    print(f"Per-scene CSV saved to {scene_csv_file}")

    out_path = results_path if inplace else (output_dir / "results_postprocessed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    print(f"Postprocessed results saved to {out_path}")


@dataclass
class EpisodeData:
    """Single episode data"""
    episode_idx: int
    instruction: str
    scene_id: str
    target_category: str
    direction: str
    goal: np.ndarray
    start: np.ndarray
    image: np.ndarray
    gt_trajectory: np.ndarray
    obstacle_mask: Optional[np.ndarray] = None  # (H, W) binary mask, 1=obstacle


class DPZarrDataset:
    """Load unified VLA dataset generated by generate_vla_v2.
    
    Dataset structure (from zarr_generator.py):
        dataset.zarr/
        ├── data/
        │   ├── img          # (N_images, 224, 224, 3) uint8
        │   ├── mask         # (N_images, 50, 50) bool - walkable mask
        │   ├── sdf          # (N_images, 50, 50) float16
        │   ├── label_mask   # (N_images, 50, 50) int16
        │   ├── state        # (N_steps, 2) float32
        │   ├── action       # (N_steps, 2) float32
        │   ├── agent_pos    # (N_steps, 2) float32
        │   └── sample_idx   # (N_steps,) int64 - maps each step to image index
        └── meta/
            └── episode_ends # (N_episodes,) int64
        episode_meta.json    # List of episode metadata
        sample_indices.json  # {sample_id: img_index} mapping
    """
    
    def __init__(self, zarr_path: str, load_mask: bool = True):
        self.zarr_path = Path(zarr_path)
        self.load_mask = load_mask
        
        # Handle both dataset.zarr subdirectory and direct path
        if (self.zarr_path / 'dataset.zarr').exists():
            self.zarr_root = self.zarr_path / 'dataset.zarr'
        else:
            self.zarr_root = self.zarr_path
        
        self.root = zarr.open_group(str(self.zarr_root), mode='r')
        
        # Load episode metadata
        episode_meta_path = self.zarr_path / 'episode_meta.json'
        if not episode_meta_path.exists():
            episode_meta_path = self.zarr_root.parent / 'episode_meta.json'
        with open(episode_meta_path, 'r') as f:
            self.episode_meta = json.load(f)
        
        # Load sample_id to image index mapping
        sample_indices_path = self.zarr_path / 'sample_indices.json'
        if not sample_indices_path.exists():
            sample_indices_path = self.zarr_root.parent / 'sample_indices.json'
        
        if sample_indices_path.exists():
            with open(sample_indices_path, 'r') as f:
                self.sample_id_to_img_idx = json.load(f)
            print(f"  Loaded sample_indices.json: {len(self.sample_id_to_img_idx)} mappings")
        else:
            # Fallback: build mapping from episode_meta (legacy compatibility)
            sample_ids = [m['sample_id'] for m in self.episode_meta]
            unique_sample_ids = sorted(set(sample_ids))
            self.sample_id_to_img_idx = {sid: i for i, sid in enumerate(unique_sample_ids)}
            print(f"  Built sample_id mapping from episode_meta: {len(self.sample_id_to_img_idx)} mappings")
        
        # Load data arrays
        self.images = self.root['data/img']
        self.agent_pos = self.root['data/agent_pos']
        self.episode_ends = self.root['meta/episode_ends'][:]
        
        # Check for embedded mask/sdf (generate_vla_v2 format)
        self.has_embedded_mask = 'mask' in self.root['data']
        self.has_embedded_sdf = 'sdf' in self.root['data']
        
        if self.has_embedded_mask:
            self.masks = self.root['data/mask']
            print(f"  Embedded masks: {self.masks.shape}")
        if self.has_embedded_sdf:
            self.sdfs = self.root['data/sdf']
            print(f"  Embedded SDFs: {self.sdfs.shape}")
        
        self.num_episodes = len(self.episode_meta)
        print(f"Loaded dataset: {self.num_episodes} episodes, {self.images.shape[0]} images")
    
    def _load_mask(self, sample_id: str, img_idx: int) -> Optional[np.ndarray]:
        """Load obstacle mask.
        
        Returns: (H, W) binary mask where 1=obstacle, 0=free
        """
        if not self.has_embedded_mask:
            return None
        
        try:
            # Embedded mask from generate_vla_v2: walkable mask (True=walkable, False=obstacle)
            mask_small = np.array(self.masks[img_idx])
            # Convert to obstacle mask: 1=obstacle, 0=free
            obstacle_mask = (~mask_small.astype(bool)).astype(np.uint8)
            return obstacle_mask
        except Exception as e:
            print(f"  Warning: Failed to load mask for {sample_id}: {e}")
            return None
    
    def get_episode(self, idx: int) -> EpisodeData:
        meta = self.episode_meta[idx]
        start_idx = 0 if idx == 0 else self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        
        sample_id = meta['sample_id']
        img_idx = self.sample_id_to_img_idx[sample_id]
        image = np.array(self.images[img_idx])
        gt_traj = np.array(self.agent_pos[start_idx:end_idx])
        
        # Load mask
        obstacle_mask = None
        if self.load_mask:
            obstacle_mask = self._load_mask(sample_id, img_idx)
        
        return EpisodeData(
            episode_idx=idx,
            instruction=meta['instruction'],
            scene_id=meta['scene_id'],
            target_category=meta['target_category'],
            direction=meta['direction'],
            goal=np.array(meta['goal']),
            start=np.array(meta['start']),
            image=image,
            gt_trajectory=gt_traj,
            obstacle_mask=obstacle_mask
        )

def evaluate_single_episode(
    episode: EpisodeData,
    *,
    grid_size: int,
    num_points: int,
    allow_diagonal: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "episode_idx": episode.episode_idx,
        "scene_id": episode.scene_id,
        "instruction": episode.instruction,
        "target_category": episode.target_category,
        "direction": episode.direction,
        "has_mask": episode.obstacle_mask is not None,
        "planner_success": False,
        "error": None,
        "vlm_target_center": None,
        "vlm_goal": None,
        "target_detection_error": None,
        "num_obstacles": 0,
        "fge": float("inf"),
        "cr": 0.0,
        "plr": 1.0,
        "curv": 0.0,
        "pred_traj_len": 0,
        "gt_traj_len": int(len(episode.gt_trajectory)),
    }

    try:
        vlm_planner = get_vlm_planner()
        path_planner = get_path_planner()

        vlm = vlm_planner.analyze_scene(episode.image, episode.instruction, add_grid=False, require_target=True)
        if not isinstance(vlm, dict) or vlm.get("error"):
            raise ValueError(str(vlm.get("error") if isinstance(vlm, dict) else "VLM returned non-dict"))

        target = vlm.get("target")
        if not isinstance(target, dict):
            raise ValueError("Missing target")

        if isinstance(target.get("center"), (list, tuple)) and len(target.get("center")) == 2:
            result["vlm_target_center"] = [float(target["center"][0]), float(target["center"][1])]

        goal_obj = vlm.get("goal")
        if not (isinstance(goal_obj, dict) and goal_obj.get("x") is not None and goal_obj.get("y") is not None):
            raise ValueError("Missing goal")

        goal_xy = (float(goal_obj["x"]), float(goal_obj["y"]))
        result["vlm_goal"] = [float(goal_xy[0]), float(goal_xy[1])]
        result["target_detection_error"] = float(np.linalg.norm(np.array(goal_xy) - episode.goal))

        obstacles = vlm.get("obstacles") or []
        obstacle_bboxes = []
        if isinstance(obstacles, list):
            for obs in obstacles:
                if isinstance(obs, dict) and isinstance(obs.get("bbox"), (list, tuple)) and len(obs.get("bbox")) == 4:
                    obstacle_bboxes.append([float(x) for x in obs["bbox"]])
        result["num_obstacles"] = int(len(obstacle_bboxes))

        forbidden_bboxes = []
        if vlm.get("approach_side") and vlm.get("approach_side") != "none":
            if isinstance(target.get("bbox"), (list, tuple)) and len(target.get("bbox")) == 4:
                forbidden_bboxes.append([float(x) for x in target["bbox"]])

        path = path_planner.generate_safe_path_astar(
            start_xy=episode.start,
            goal_xy=goal_xy,
            obstacle_bboxes=obstacle_bboxes,
            forbidden_bboxes=forbidden_bboxes,
            grid_size=grid_size,
            allow_diagonal=allow_diagonal,
            num_points=num_points,
        )

        trajectory = np.array(path, dtype=float)
        result["pred_traj_len"] = int(len(trajectory))
        result["planner_success"] = True

        metrics = TrajectoryMetrics(
            pred_traj=trajectory,
            gt_traj=episode.gt_trajectory,
            goal_pos=episode.goal,
            obstacle_mask=episode.obstacle_mask,
        )

        result["fge"] = metrics.final_goal_error(num_points=STANDARD_NUM_WAYPOINTS)
        result["cr"] = metrics.collision_rate(num_points=STANDARD_NUM_WAYPOINTS)
        result["plr"] = metrics.path_length_ratio(num_points=STANDARD_NUM_WAYPOINTS)
        result["curv"] = metrics.curvature(num_points=STANDARD_NUM_WAYPOINTS)

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM+Planner (minimal, multi-threaded)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/media/dragon_llm/linux_ssd/vla_dataset_unified_static_v13/val",
        help="Path to dataset root (contains episode_meta.json and dataset.zarr or direct zarr)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vlm_planner_results",
        help="Output directory",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes (default: all)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every N-th episode (default: 1, i.e., evaluate all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Parallel workers",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=128,
        help="A* grid resolution",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50,
        help="Number of points in planned trajectory",
    )
    parser.add_argument(
        "--no_diagonal",
        action="store_true",
        help="Disable diagonal moves in A*",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Postprocess an existing results.json under output_dir (no evaluation run)",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help="Path to results.json for postprocess (default: output_dir/results.json)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input results.json when postprocessing (default: write results_postprocessed.json)",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.postprocess:
        results_path = Path(args.results_json) if args.results_json else (output_dir / "results.json")
        postprocess_existing_results(results_path=results_path, output_dir=output_dir, inplace=bool(args.inplace))
        return

    print(f"Loading dataset from {args.dataset}")
    dataset = DPZarrDataset(args.dataset)

    num_episodes_limit = args.num_episodes or dataset.num_episodes
    num_episodes_limit = min(num_episodes_limit, dataset.num_episodes)

    stride = max(1, int(args.stride))
    episode_indices = list(range(0, num_episodes_limit, stride))
    num_eval = len(episode_indices)

    allow_diagonal = not args.no_diagonal
    results: List[Dict[str, Any]] = []

    print(
        f"Evaluating {num_eval} episodes (stride={stride}, limit={num_episodes_limit}) "
        f"with {args.num_workers} workers"
    )

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_idx = {}
        for idx in episode_indices:
            episode = dataset.get_episode(idx)
            fut = executor.submit(
                evaluate_single_episode,
                episode,
                grid_size=args.grid_size,
                num_points=args.num_points,
                allow_diagonal=allow_diagonal,
            )
            future_to_idx[fut] = idx

        with tqdm(total=num_eval, desc="Episodes", dynamic_ncols=True) as pbar:
            for fut in as_completed(future_to_idx):
                r = fut.result()
                results.append(r)
                pbar.update(1)

    # Sort results by episode_idx for stable outputs.
    results.sort(key=lambda x: int(x.get("episode_idx", 0)))

    valid = [r for r in results if not r.get("error") and r.get("fge") != float("inf")]
    err_count = sum(1 for r in results if r.get("error"))
    mask_count = sum(1 for r in valid if r.get("has_mask"))

    def _mean(xs: List[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    def _std(xs: List[float]) -> Optional[float]:
        return float(np.std(xs)) if xs else None

    fge_vals = [float(r["fge"]) for r in valid]
    cr_vals = [float(r["cr"]) for r in valid]
    plr_vals = [float(r["plr"]) for r in valid]
    curv_vals = [float(r["curv"]) for r in valid]
    tgt_errs = [float(r["target_detection_error"]) for r in valid if r.get("target_detection_error") is not None]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(results)}")
    print(f"Valid episodes: {len(valid)}")
    print(f"Errors: {err_count}")
    print(f"Valid episodes with mask: {mask_count}/{len(valid)}")
    if valid:
        print(f"FGE  : {_mean(fge_vals):.4f} ± {_std(fge_vals):.4f}")
        print(f"CR   : {_mean(cr_vals)*100.0:.2f}%")
        print(f"PLR  : {_mean(plr_vals):.4f} ± {_std(plr_vals):.4f}")
        print(f"Curv : {_mean(curv_vals):.4f} ± {_std(curv_vals):.4f}")
    if tgt_errs:
        print(f"Target error: {_mean(tgt_errs):.4f} ± {_std(tgt_errs):.4f}")

    results_file = output_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "dataset": args.dataset,
                    "num_episodes": num_eval,
                    "num_episodes_limit": int(num_episodes_limit),
                    "stride": int(stride),
                    "num_workers": args.num_workers,
                    "grid_size": args.grid_size,
                    "num_points": args.num_points,
                    "allow_diagonal": allow_diagonal,
                },
                "overall": {
                    "fge_mean": _mean(fge_vals),
                    "fge_std": _std(fge_vals),
                    "cr_mean": _mean(cr_vals),
                    "plr_mean": _mean(plr_vals),
                    "plr_std": _std(plr_vals),
                    "curv_mean": _mean(curv_vals),
                    "curv_std": _std(curv_vals),
                    "target_error_mean": _mean(tgt_errs),
                    "target_error_std": _std(tgt_errs),
                    "num_valid": len(valid),
                    "num_errors": err_count,
                    "num_with_mask": mask_count,
                },
                "episode_results": results,
                "per_scene": compute_per_scene_summary(results),
            },
            f,
            indent=2,
        )
    print(f"Results saved to {results_file}")

    scene_csv_file = output_dir / "scene_metrics.csv"
    per_scene = compute_per_scene_summary(results)
    with open(scene_csv_file, "w", encoding="utf-8") as f:
        f.write("scene_id,num_episodes,fge,fge_std,cr,cr_std,plr,plr_std,curv,curv_std,target_error,target_error_std\n")
        for s in per_scene:
            f.write(
                f"{s.get('scene_id','')},{s.get('num_episodes','')},"
                f"{s.get('fge','')},{s.get('fge_std','')},"
                f"{s.get('cr','')},{s.get('cr_std','')},"
                f"{s.get('plr','')},{s.get('plr_std','')},"
                f"{s.get('curv','')},{s.get('curv_std','')},"
                f"{s.get('target_error','')},{s.get('target_error_std','')}\n"
            )
    print(f"Per-scene CSV saved to {scene_csv_file}")

    csv_file = output_dir / "metrics.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(
            "episode_idx,scene_id,fge,cr,plr,curv,target_error,num_obstacles,pred_traj_len,gt_traj_len,error\n"
        )
        for r in results:
            f.write(
                f"{r.get('episode_idx','')},{r.get('scene_id','')},{r.get('fge','')},{r.get('cr','')},{r.get('plr','')},{r.get('curv','')},"
                f"{r.get('target_detection_error','')},{r.get('num_obstacles','')},{r.get('pred_traj_len','')},{r.get('gt_traj_len','')},"
                f"{str(r.get('error','')).replace(',', ' ')}\n"
            )
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
