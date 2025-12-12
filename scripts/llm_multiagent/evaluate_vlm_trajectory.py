#!/usr/bin/env python3
"""
VLM Direct Trajectory Evaluation Script (Multi-threaded)

This script evaluates the pure VLM trajectory generation baseline.
Unlike VLM+Planner, this approach directly asks the VLM to generate
a complete collision-free trajectory from start to goal.

Comparison:
- VLM+Planner: VLM detects target → Traditional planner generates path
- VLM Trajectory: VLM directly generates the complete trajectory

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

# Import VLM trajectory planner
from vlm_trajectory_planner import VLMTrajectoryPlanner

# Thread-local storage for planners (each thread gets its own instance)
thread_local = threading.local()


def get_vlm_trajectory_planner(num_waypoints: int = 16):
    """Get thread-local VLM trajectory planner instance"""
    if not hasattr(thread_local, 'vlm_traj_planner'):
        thread_local.vlm_traj_planner = VLMTrajectoryPlanner(num_waypoints=num_waypoints)
    return thread_local.vlm_traj_planner


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
    obstacle_mask: Optional[np.ndarray] = None


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


def evaluate_single_episode(episode: EpisodeData, num_waypoints: int = 16,
                           max_retries: int = 3) -> Dict[str, Any]:
    """
    Evaluate a single episode using VLM direct trajectory generation
    VLM only receives: image + instruction + start position (NO goal)
    """
    result = {
        'episode_idx': episode.episode_idx,
        'scene_id': episode.scene_id,
        'instruction': episode.instruction,
        'target_category': episode.target_category,
        'direction': episode.direction,
        'fge': float('inf'),
        'cr': 0.0,
        'plr': 1.0,
        'curv': 0.0,
        'target_detection_error': None,  # Error between VLM-detected target and GT goal
        'vlm_target': None,
        'vlm_reasoning': None,
        'vlm_error': None,
        'planner_success': False,
        'error': None,
        'retries': 0,
        'has_mask': episode.obstacle_mask is not None
    }
    
    for attempt in range(max_retries):
        try:
            # Get thread-local planner
            planner = get_vlm_trajectory_planner(num_waypoints)
            
            # Only use GT start - VLM must identify goal from instruction
            start = episode.start
            
            # Generate trajectory directly with VLM (NO goal provided)
            vlm_result = planner.generate_trajectory(
                image=episode.image,
                instruction=episode.instruction,
                start=start,
                add_grid=True
            )
            
            result['vlm_reasoning'] = vlm_result.get('reasoning', '')
            result['vlm_error'] = vlm_result.get('error')
            
            # Get VLM-identified target
            if vlm_result.get('target'):
                vlm_target = vlm_result['target']
                result['vlm_target'] = [vlm_target['x'], vlm_target['y']]
                # Calculate target detection error (how well VLM understood the instruction)
                pred_target = np.array([vlm_target['x'], vlm_target['y']])
                result['target_detection_error'] = float(np.linalg.norm(pred_target - episode.goal))
            
            # Get trajectory
            trajectory = np.array(vlm_result['trajectory'])
            
            if len(trajectory) == 0:
                raise ValueError("Empty trajectory returned")
            
            result['planner_success'] = True
            
            # Compute metrics
            metrics = TrajectoryMetrics(
                pred_traj=trajectory,
                gt_traj=episode.gt_trajectory,
                goal_pos=episode.goal,
                obstacle_mask=episode.obstacle_mask
            )
            
            result['fge'] = metrics.final_goal_error()
            result['cr'] = metrics.collision_rate()
            result['plr'] = metrics.path_length_ratio()
            result['curv'] = metrics.curvature()
            result['pred_traj_len'] = len(trajectory)
            result['gt_traj_len'] = len(episode.gt_trajectory)
            result['retries'] = attempt
            
            # Store trajectory for visualization
            result['pred_trajectory'] = trajectory.tolist()
            
            return result
            
        except Exception as e:
            result['retries'] = attempt + 1
            result['error'] = str(e)
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                traceback.print_exc()
    
    return result


def visualize_episode(episode: EpisodeData, result: Dict[str, Any], 
                     output_path: Path):
    """Visualize episode with predicted and GT trajectories"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Show image
    ax.imshow(episode.image)
    
    h, w = episode.image.shape[:2]
    
    # Plot GT trajectory (blue)
    gt_traj = episode.gt_trajectory
    gt_px = gt_traj * np.array([w, h])
    ax.plot(gt_px[:, 0], gt_px[:, 1], 'b-', linewidth=2, label='GT', alpha=0.7)
    ax.scatter(gt_px[0, 0], gt_px[0, 1], c='blue', s=100, marker='o', zorder=5)
    ax.scatter(gt_px[-1, 0], gt_px[-1, 1], c='blue', s=100, marker='*', zorder=5)
    
    # Plot predicted trajectory (red)
    if result.get('pred_trajectory'):
        pred_traj = np.array(result['pred_trajectory'])
        pred_px = pred_traj * np.array([w, h])
        ax.plot(pred_px[:, 0], pred_px[:, 1], 'r-', linewidth=2, label='VLM Pred', alpha=0.7)
        ax.scatter(pred_px[0, 0], pred_px[0, 1], c='red', s=100, marker='o', zorder=5)
        ax.scatter(pred_px[-1, 0], pred_px[-1, 1], c='red', s=100, marker='*', zorder=5)
    
    # Plot start and goal markers
    start_px = episode.start * np.array([w, h])
    goal_px = episode.goal * np.array([w, h])
    ax.scatter(start_px[0], start_px[1], c='green', s=200, marker='s', 
              label='Start', zorder=10, edgecolors='black', linewidths=2)
    ax.scatter(goal_px[0], goal_px[1], c='yellow', s=200, marker='D', 
              label='Goal', zorder=10, edgecolors='black', linewidths=2)
    
    # Title with metrics
    title = f"Episode {episode.episode_idx}: {episode.instruction[:50]}..."
    title += f"\nFGE={result.get('fge', 0):.4f}, CR={result.get('cr', 0)*100:.0f}%, "
    title += f"PLR={result.get('plr', 1):.2f}, Curv={result.get('curv', 0):.3f}"
    ax.set_title(title, fontsize=10)
    
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(results: List[Dict], output_dir: Path, 
                   checkpoint_name: str = "checkpoint.json"):
    """Save intermediate results"""
    checkpoint_path = output_dir / checkpoint_name
    # Remove pred_trajectory from checkpoint to save space
    results_to_save = []
    for r in results:
        r_copy = r.copy()
        r_copy.pop('pred_trajectory', None)
        results_to_save.append(r_copy)
    with open(checkpoint_path, 'w') as f:
        json.dump(results_to_save, f)


def load_checkpoint(output_dir: Path, 
                   checkpoint_name: str = "checkpoint.json") -> List[Dict]:
    """Load checkpoint if exists"""
    checkpoint_path = output_dir / checkpoint_name
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM Direct Trajectory (Multi-threaded)')
    parser.add_argument('--dataset', type=str, 
                       default='/media/dragon_llm/linux_ssd/vla_dataset_unified/val',
                       help='Path to Zarr dataset')
    parser.add_argument('--output_dir', type=str,
                       default='vlm_trajectory_results_v2   ',
                       help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=None,   
                       help='Number of episodes (None = all)')
    parser.add_argument('--num_waypoints', type=int, default=5,
                       help='Number of waypoints to generate')
    parser.add_argument('--num_workers', type=int, default=4,
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
    
    if args.visualize_every > 0:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
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
    
    print(f"\n{'='*60}")
    print("VLM DIRECT TRAJECTORY EVALUATION")
    print(f"{'='*60}")
    print(f"Method: VLM generates {args.num_waypoints} waypoints directly")
    print(f"Episodes: {len(episodes_to_process)} remaining")
    print(f"Workers: {args.num_workers}")
    print(f"Max retries: {args.max_retries}")
    print(f"{'='*60}\n")
    
    # Process with thread pool
    processed_count = len(completed_indices)
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx in episodes_to_process:
            episode = dataset.get_episode(idx)
            future = executor.submit(
                evaluate_single_episode, 
                episode, 
                args.num_waypoints,
                args.max_retries
            )
            future_to_idx[future] = idx
        
        # Process completed tasks with progress bar
        with tqdm(total=len(episodes_to_process), initial=0) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Count as error only if planning actually failed (not just had retries)
                    if not result.get('planner_success', False):
                        error_count += 1
                    
                    processed_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'errors': error_count,
                        'fge': f"{result.get('fge', 0):.3f}" if result.get('fge') != float('inf') else 'inf',
                        'cr': f"{result.get('cr', 0)*100:.0f}%"
                    })
                    
                    # Visualize if requested
                    if args.visualize_every > 0 and idx % args.visualize_every == 0:
                        episode = dataset.get_episode(idx)
                        vis_path = output_dir / "visualizations" / f"episode_{idx:05d}.png"
                        visualize_episode(episode, result, vis_path)
                    
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
    valid_results = [r for r in results if r.get('fge') != float('inf')]
    
    fge_values = [r['fge'] for r in valid_results]
    cr_values = [r['cr'] for r in valid_results]
    plr_values = [r['plr'] for r in valid_results]
    curv_values = [r['curv'] for r in valid_results]
    target_errors = [r['target_detection_error'] for r in valid_results 
                    if r.get('target_detection_error') is not None]
    mask_count = sum(1 for r in valid_results if r.get('has_mask', False))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY - VLM DIRECT TRAJECTORY")
    print("=" * 60)
    print(f"Total episodes: {len(results)}")
    print(f"Valid episodes: {len(valid_results)}")
    print(f"Failed episodes: {error_count}")
    print(f"Episodes with mask: {mask_count}/{len(valid_results)}")
    print(f"\nTrajectory Metrics:")
    print(f"  FGE  (Final Goal Error):  {np.mean(fge_values):.4f} ± {np.std(fge_values):.4f}")
    print(f"  CR   (Collision Rate):    {np.mean(cr_values)*100:.2f}%")
    print(f"  PLR  (Path Length Ratio): {np.mean(plr_values):.4f} ± {np.std(plr_values):.4f}")
    print(f"  Curv (Curvature):         {np.mean(curv_values):.4f} ± {np.std(curv_values):.4f}")
    
    # Target detection metrics (how well VLM understands instruction)
    if target_errors:
        print(f"\nVLM Target Detection (instruction understanding):")
        print(f"  Target Error:    {np.mean(target_errors):.4f} ± {np.std(target_errors):.4f}")
        print(f"  Detection Rate:  {len(target_errors)/len(valid_results)*100:.1f}%")
    
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
        scene_fge = np.mean([r['fge'] for r in scene_res])
        scene_cr = np.mean([r['cr'] for r in scene_res])
        scene_curv = np.mean([r['curv'] for r in scene_res])
        print(f"{scene_id}: FGE={scene_fge:.4f}, CR={scene_cr*100:.1f}%, Curv={scene_curv:.4f}, N={len(scene_res)}")
        scene_summary.append({
            'scene_id': scene_id,
            'fge': float(scene_fge),
            'cr': float(scene_cr),
            'curv': float(scene_curv),
            'num_episodes': len(scene_res)
        })
    
    # Save final results
    results_file = output_dir / "results.json"
    
    # Clean results for saving (remove large trajectory data)
    results_to_save = []
    for r in results:
        r_copy = r.copy()
        r_copy.pop('pred_trajectory', None)
        results_to_save.append(r_copy)
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'method': 'VLM Direct Trajectory (no goal input)',
                'description': 'VLM receives image + instruction + start only, must identify target',
                'dataset': args.dataset,
                'num_episodes': len(results),
                'num_waypoints': args.num_waypoints,
                'num_workers': args.num_workers,
                'max_retries': args.max_retries
            },
            'overall': {
                'fge_mean': float(np.mean(fge_values)),
                'fge_std': float(np.std(fge_values)),
                'cr_mean': float(np.mean(cr_values)),
                'plr_mean': float(np.mean(plr_values)),
                'plr_std': float(np.std(plr_values)),
                'curv_mean': float(np.mean(curv_values)),
                'curv_std': float(np.std(curv_values)),
                'target_detection_error_mean': float(np.mean(target_errors)) if target_errors else None,
                'target_detection_error_std': float(np.std(target_errors)) if target_errors else None,
                'target_detection_rate': len(target_errors)/len(valid_results) if valid_results else 0,
                'num_valid': len(valid_results),
                'num_errors': error_count,
                'num_with_mask': mask_count
            },
            'per_scene': scene_summary,
            'episode_results': results_to_save
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save CSV
    csv_file = output_dir / "metrics.csv"
    with open(csv_file, 'w') as f:
        f.write("episode_idx,scene_id,instruction,fge,cr,plr,curv,target_error,vlm_target,vlm_error,error\n")
        for r in results:
            instruction = r.get('instruction', '')[:50].replace(',', ' ').replace('\n', ' ')
            vlm_target = r.get('vlm_target', '')
            if vlm_target:
                vlm_target = f"({vlm_target[0]:.3f};{vlm_target[1]:.3f})"
            f.write(f"{r.get('episode_idx', '')},{r.get('scene_id', '')},{instruction},"
                   f"{r.get('fge', '')},{r.get('cr', '')},{r.get('plr', '')},{r.get('curv', '')},"
                   f"{r.get('target_detection_error', '')},{vlm_target},"
                   f"{r.get('vlm_error', '')},{r.get('error', '')}\n")
    
    print(f"CSV saved to {csv_file}")
    
    # Print comparison hint
    print("\n" + "=" * 60)
    print("COMPARISON WITH OTHER METHODS")
    print("=" * 60)
    print("To compare with VLM+Planner baseline, run:")
    print(f"  python evaluate_vlm_planner.py --dataset {args.dataset}")
    print("\nTo compare with VLA model, run:")
    print("  python -m diffusion_policy.scripts.test.run_test --checkpoint <path>")


if __name__ == "__main__":
    main()
