#!/usr/bin/env python3
"""
VLM + Planner Evaluation Script for Semantic Map Dataset (Multi-threaded)

This script evaluates the VLM+Planner baseline on the VLA dataset.
Features:
- Multi-threaded VLM calls for faster evaluation
- Automatic retry on errors
- Checkpoint saving for resuming interrupted runs
"""

import os
import sys
import json
import zarr
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for threading
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

# Add paths for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "planners"))

# Import planners
from path_planner import PathPlanner
from semantic_map_planner import SemanticMapPlanner

# Thread-local storage for planners (each thread gets its own instance)
thread_local = threading.local()


def get_vlm_planner():
    """Get thread-local VLM planner instance"""
    if not hasattr(thread_local, 'vlm_planner'):
        thread_local.vlm_planner = SemanticMapPlanner()
    return thread_local.vlm_planner


def get_path_planner():
    """Get thread-local path planner instance"""
    if not hasattr(thread_local, 'path_planner'):
        thread_local.path_planner = PathPlanner(num_ctrl_points=8, obstacle_effect_area=0.5)
    return thread_local.path_planner


# Standard number of waypoints for fair curvature comparison
STANDARD_NUM_WAYPOINTS = 20


def resample_trajectory(traj: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample trajectory to fixed number of points using linear interpolation.
    This ensures fair comparison of curvature across different methods.
    
    Args:
        traj: Original trajectory [N, 2]
        num_points: Target number of points
    
    Returns:
        Resampled trajectory [num_points, 2]
    """
    if len(traj) < 2:
        return traj
    
    if len(traj) == num_points:
        return traj
    
    # Compute cumulative arc length
    diffs = np.diff(traj, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < 1e-8:
        # Degenerate trajectory (all points same)
        return np.tile(traj[0], (num_points, 1))
    
    # Generate uniform samples along arc length
    target_lengths = np.linspace(0, total_length, num_points)
    
    # Interpolate
    resampled = np.zeros((num_points, 2))
    for i, target_len in enumerate(target_lengths):
        # Find segment containing this length
        idx = np.searchsorted(cumulative_length, target_len, side='right') - 1
        idx = np.clip(idx, 0, len(traj) - 2)
        
        # Interpolate within segment
        seg_start_len = cumulative_length[idx]
        seg_len = segment_lengths[idx] if idx < len(segment_lengths) else 1e-8
        
        if seg_len < 1e-8:
            t = 0
        else:
            t = (target_len - seg_start_len) / seg_len
        t = np.clip(t, 0, 1)
        
        resampled[i] = traj[idx] * (1 - t) + traj[idx + 1] * t
    
    return resampled


class TrajectoryMetrics:
    """
    Trajectory evaluation metrics (same as HMRS/scripts/test/metrics.py)
    - FGE: Final Goal Error (Euclidean distance to goal)
    - CR: Collision Rate (1 if any point hits obstacle, 0 otherwise)
    - PLR: Path Length Ratio (pred_length / gt_length)
    - Curv: Curvature (mean absolute angle change between segments)
    
    Note: Curvature is computed on resampled trajectory (STANDARD_NUM_WAYPOINTS points)
    for fair comparison across methods with different waypoint counts.
    """
    def __init__(self, pred_traj: np.ndarray, gt_traj: np.ndarray, 
                 goal_pos: np.ndarray, obstacle_mask: Optional[np.ndarray] = None):
        self.pred_traj = np.array(pred_traj)
        self.gt_traj = np.array(gt_traj)
        self.goal_pos = np.array(goal_pos)
        self.obstacle_mask = obstacle_mask  # (H, W) binary mask, 1=obstacle
        
    def final_goal_error(self) -> float:
        """FGE: Euclidean distance from final position to goal"""
        if len(self.pred_traj) == 0:
            return float('inf')
        return float(np.linalg.norm(self.pred_traj[-1] - self.goal_pos))
    
    def collision_rate(self) -> float:
        """CR: 1.0 if trajectory collides with obstacle, 0.0 otherwise"""
        if self.obstacle_mask is None:
            return 0.0
        H, W = self.obstacle_mask.shape
        for pt in self.pred_traj:
            # Convert normalized [0,1] to pixel coordinates
            cx = int(pt[0] * W)
            cy = int(pt[1] * H)
            # Clamp to valid range
            cx = np.clip(cx, 0, W - 1)
            cy = np.clip(cy, 0, H - 1)
            if self.obstacle_mask[cy, cx] == 1:
                return 1.0
        return 0.0
    
    def path_length_ratio(self) -> float:
        """PLR: pred_path_length / gt_path_length"""
        if len(self.pred_traj) < 2 or len(self.gt_traj) < 2:
            return 1.0
        pred_len = self._compute_path_length(self.pred_traj)
        gt_len = self._compute_path_length(self.gt_traj)
        return float(pred_len / gt_len) if gt_len > 1e-6 else 1.0
    
    def curvature(self, num_points: int = STANDARD_NUM_WAYPOINTS) -> float:
        """Curv: Mean absolute angle change between consecutive segments (radians)
        
        Note: Trajectory is resampled to num_points for fair comparison.
        """
        # Resample to standard number of points for fair comparison
        resampled_traj = resample_trajectory(self.pred_traj, num_points)
        return self._compute_curvature(resampled_traj)
    
    def _compute_path_length(self, path: np.ndarray) -> float:
        """Compute total path length"""
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    
    def _compute_curvature(self, path: np.ndarray) -> float:
        """Compute mean curvature as angle change between segments"""
        if len(path) < 3:
            return 0.0
        # Compute direction vectors
        vectors = path[1:] - path[:-1]
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-6
        vectors = vectors[valid]
        if len(vectors) < 2:
            return 0.0
        # Compute angles
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        # Compute angle differences, wrapped to [-pi, pi]
        diffs = angles[1:] - angles[:-1]
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        return float(np.mean(np.abs(diffs)))


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


def evaluate_single_episode(episode: EpisodeData, max_retries: int = 3) -> Dict[str, Any]:
    """
    Evaluate a single episode with retry mechanism
    """
    result = {
        'episode_idx': episode.episode_idx,
        'scene_id': episode.scene_id,
        'instruction': episode.instruction,
        'target_category': episode.target_category,
        'direction': episode.direction,
        'fge': float('inf'),
        'cr': 0.0,  # Collision Rate (1=collision, 0=no collision)
        'plr': 1.0,
        'curv': 0.0,
        'target_detection_error': None,
        'vlm_target': None,
        'num_obstacles': 0,
        'planner_success': False,
        'error': None,
        'retries': 0,
        'has_mask': episode.obstacle_mask is not None
    }
    
    for attempt in range(max_retries):
        try:
            # Get thread-local planners
            vlm_planner = get_vlm_planner()
            path_planner = get_path_planner()
            
            # Always use GT start
            start = episode.start
            
            # Call VLM to detect target and obstacles
            vlm_result = vlm_planner.analyze_scene(episode.image, episode.instruction)
            
            # Get target from VLM
            if vlm_result.get('target') and vlm_result['target'].get('x') is not None:
                goal = np.array([vlm_result['target']['x'], vlm_result['target']['y']])
                result['vlm_target'] = [float(goal[0]), float(goal[1])]
                result['target_detection_error'] = float(np.linalg.norm(goal - episode.goal))
            else:
                # VLM failed to detect target - this is a failure case
                # Use GT goal for path planning but mark metrics as failed
                goal = episode.goal
                result['vlm_target'] = None
                result['vlm_target_detection_failed'] = True
            
            # Get obstacles from VLM
            obstacles = []
            if vlm_result.get('obstacles'):
                for obs in vlm_result['obstacles']:
                    if obs.get('x') is not None:
                        obstacles.append(np.array([obs['x'], obs['y']]))
            result['num_obstacles'] = len(obstacles)
            
            # Generate path using PathPlanner
            scale = 10.0
            start_scaled = start * scale
            goal_scaled = goal * scale
            obstacles_scaled = [(obs * scale).tolist() for obs in obstacles]
            
            path = path_planner.generate_safe_path(
                A=start_scaled.tolist(),
                C=goal_scaled.tolist(),
                obstacles=obstacles_scaled
            )
            result['planner_success'] = True
            
            # Convert back to normalized coordinates
            trajectory = np.array([[p[0] / scale, p[1] / scale] for p in path])
            
            # Compute metrics with obstacle mask
            metrics = TrajectoryMetrics(
                pred_traj=trajectory,
                gt_traj=episode.gt_trajectory,
                goal_pos=episode.goal,
                obstacle_mask=episode.obstacle_mask  # Pass obstacle mask for CR
            )
            
            # If VLM failed to detect target, FGE should be computed against GT goal
            # but marked as invalid (since we cheated by using GT goal for planning)
            if result.get('vlm_target_detection_failed'):
                # FGE is meaningless when VLM failed - we used GT goal, so FGE≈0
                # Set to None to exclude from statistics, or compute actual error
                # which would be the distance from the path endpoint to GT goal
                result['fge'] = None  # Mark as invalid
            else:
                result['fge'] = metrics.final_goal_error()
            
            result['cr'] = metrics.collision_rate()  # Now computes real collision rate
            result['plr'] = metrics.path_length_ratio()
            result['curv'] = metrics.curvature()
            result['pred_traj_len'] = len(trajectory)
            result['gt_traj_len'] = len(episode.gt_trajectory)
            result['retries'] = attempt
            
            return result
            
        except Exception as e:
            result['retries'] = attempt + 1
            result['error'] = str(e)
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                # Final attempt failed, return with error
                traceback.print_exc()
    
    return result


def save_checkpoint(results: List[Dict], output_dir: Path, checkpoint_name: str = "checkpoint.json"):
    """Save intermediate results"""
    checkpoint_path = output_dir / checkpoint_name
    with open(checkpoint_path, 'w') as f:
        json.dump(results, f)


def load_checkpoint(output_dir: Path, checkpoint_name: str = "checkpoint.json") -> List[Dict]:
    """Load checkpoint if exists"""
    checkpoint_path = output_dir / checkpoint_name
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM+Planner (Multi-threaded)')
    parser.add_argument('--dataset', type=str, 
                       default='/media/dragon_llm/linux_ssd/vla_dataset_unified/val',
                       help='Path to Zarr dataset')
    parser.add_argument('--output_dir', type=str,
                       default='vlm_planner_results_v2',
                       help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of episodes (None = all)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of parallel workers')
    parser.add_argument('--max_retries', type=int, default=10,
                       help='Max retries per episode')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--visualize_every', type=int, default=0,
                       help='Visualize every N episodes (0 = disabled)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = DPZarrDataset(args.dataset)
    
    # Determine episodes to evaluate
    num_episodes = args.num_episodes or dataset.num_episodes
    num_episodes = min(num_episodes, dataset.num_episodes)
    
    # Load checkpoint if resuming
    results = []
    completed_indices = set()
    if args.resume:
        results = load_checkpoint(output_dir)
        completed_indices = {r['episode_idx'] for r in results}
        print(f"Resumed from checkpoint: {len(completed_indices)} episodes completed")
    
    # Get episodes to process
    episodes_to_process = [i for i in range(num_episodes) if i not in completed_indices]
    
    print(f"\nEvaluating {len(episodes_to_process)} episodes with {args.num_workers} workers...")
    print(f"Max retries: {args.max_retries}, Checkpoint every: {args.checkpoint_every}")
    
    # Process with thread pool
    processed_count = len(completed_indices)
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx in episodes_to_process:
            episode = dataset.get_episode(idx)
            future = executor.submit(evaluate_single_episode, episode, args.max_retries)
            future_to_idx[future] = idx
        
        # Process completed tasks with progress bar
        with tqdm(total=len(episodes_to_process), initial=0) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('error'):
                        error_count += 1
                    
                    processed_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'errors': error_count,
                        'fge': f"{result.get('fge', 0):.3f}" if result.get('fge') != float('inf') else 'inf'
                    })
                    
                    # Save checkpoint
                    if processed_count % args.checkpoint_every == 0:
                        save_checkpoint(results, output_dir)
                        
                except Exception as e:
                    print(f"\nFatal error on episode {idx}: {e}")
                    error_count += 1
                    results.append({
                        'episode_idx': idx,
                        'error': str(e),
                        'fge': float('inf'),
                        'cr': 0.0,
                        'plr': 1.0,
                        'curv': 0.0
                    })
    
    # Final checkpoint
    save_checkpoint(results, output_dir)
    
    # Compute aggregate metrics
    # Exclude inf values and None values (VLM detection failures)
    valid_results = [r for r in results if r.get('fge') is not None and r.get('fge') != float('inf')]
    # Count VLM detection failures
    vlm_detection_failures = sum(1 for r in results if r.get('vlm_target_detection_failed', False))
    
    fge_values = [r['fge'] for r in valid_results]
    cr_values = [r['cr'] for r in valid_results]
    plr_values = [r['plr'] for r in valid_results]
    curv_values = [r['curv'] for r in valid_results]
    target_errors = [r['target_detection_error'] for r in valid_results if r.get('target_detection_error') is not None]
    mask_count = sum(1 for r in valid_results if r.get('has_mask', False))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(results)}")
    print(f"Valid episodes (with VLM detection): {len(valid_results)}")
    print(f"VLM detection failures: {vlm_detection_failures}")
    print(f"Failed episodes (other errors): {error_count}")
    print(f"Episodes with mask: {mask_count}/{len(valid_results)}")
    print(f"\nTrajectory Metrics (only for episodes with successful VLM detection):")
    print(f"  FGE  (Final Goal Error):  {np.mean(fge_values):.4f} ± {np.std(fge_values):.4f}")
    print(f"  CR   (Collision Rate):    {np.mean(cr_values)*100:.2f}%")
    print(f"  PLR  (Path Length Ratio): {np.mean(plr_values):.4f} ± {np.std(plr_values):.4f}")
    print(f"  Curv (Curvature):         {np.mean(curv_values):.4f} ± {np.std(curv_values):.4f}")
    
    if target_errors:
        print(f"\nVLM Target Detection:")
        print(f"  Mean Error: {np.mean(target_errors):.4f} ± {np.std(target_errors):.4f}")
        print(f"  Detection Rate: {len(target_errors)/len(valid_results)*100:.1f}%")
    
    # Per-scene results
    scene_results = {}
    for r in valid_results:
        scene_id = r.get('scene_id', 'unknown')
        if scene_id not in scene_results:
            scene_results[scene_id] = []
        scene_results[scene_id].append(r)
    
    print("\n" + "=" * 60)
    print("PER-SCENE RESULTS")
    print("=" * 60)
    
    scene_summary = []
    for scene_id, scene_res in sorted(scene_results.items()):
        # Filter out VLM detection failures for FGE calculation
        scene_valid_fge = [r['fge'] for r in scene_res if r.get('fge') is not None]
        scene_fge = np.mean(scene_valid_fge) if scene_valid_fge else float('nan')
        scene_cr = np.mean([r['cr'] for r in scene_res])
        scene_plr = np.mean([r['plr'] for r in scene_res])
        scene_curv = np.mean([r['curv'] for r in scene_res])
        vlm_fail_count = sum(1 for r in scene_res if r.get('vlm_target_detection_failed', False))
        print(f"{scene_id}: FGE={scene_fge:.4f}, CR={scene_cr*100:.1f}%, PLR={scene_plr:.3f}, Curv={scene_curv:.4f}, N={len(scene_res)}, VLM_fail={vlm_fail_count}")
        scene_summary.append({
            'scene_id': scene_id,
            'fge': float(scene_fge) if not np.isnan(scene_fge) else None,
            'cr': float(scene_cr),
            'plr': float(scene_plr),
            'curv': float(scene_curv),
            'num_episodes': len(scene_res),
            'vlm_detection_failures': vlm_fail_count
        })
    
    # Save final results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'dataset': args.dataset,
                'num_episodes': len(results),
                'num_workers': args.num_workers,
                'max_retries': args.max_retries
            },
            'overall': {
                'fge_mean': float(np.mean(fge_values)) if fge_values else None,
                'fge_std': float(np.std(fge_values)) if fge_values else None,
                'cr_mean': float(np.mean(cr_values)) if cr_values else None,  # Collision Rate
                'plr_mean': float(np.mean(plr_values)) if plr_values else None,
                'plr_std': float(np.std(plr_values)) if plr_values else None,
                'curv_mean': float(np.mean(curv_values)) if curv_values else None,
                'curv_std': float(np.std(curv_values)) if curv_values else None,
                'target_detection_error': float(np.mean(target_errors)) if target_errors else None,
                'num_valid': len(valid_results),
                'num_vlm_detection_failures': vlm_detection_failures,
                'num_errors': error_count,
                'num_with_mask': mask_count
            },
            'per_scene': scene_summary,
            'episode_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save CSV
    csv_file = output_dir / "metrics.csv"
    with open(csv_file, 'w') as f:
        f.write("episode_idx,scene_id,instruction,fge,cr,plr,curv,target_error,num_obstacles,error\n")
        for r in results:
            instruction = r.get('instruction', '')[:50].replace(',', ' ').replace('\n', ' ')
            f.write(f"{r.get('episode_idx', '')},{r.get('scene_id', '')},{instruction},"
                   f"{r.get('fge', '')},{r.get('cr', '')},{r.get('plr', '')},{r.get('curv', '')},"
                   f"{r.get('target_detection_error', '')},{r.get('num_obstacles', '')},"
                   f"{r.get('error', '')}\n")
    
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
