#!/usr/bin/env python3
"""
Fix results.json to add PLR (Path Length Ratio) per scene

Usage:
    python fix_results_plr.py vlm_planner_full/results.json
    python fix_results_plr.py vlm_trajectory_results/results.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def fix_results(results_path: str):
    """Add PLR to per_scene statistics from episode_results"""
    
    results_path = Path(results_path)
    
    print(f"Loading {results_path}...")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Gather per-scene PLR from episode_results
    scene_plr = defaultdict(list)
    
    for ep in data.get('episode_results', []):
        scene_id = ep.get('scene_id', 'unknown')
        plr = ep.get('plr')
        if plr is not None and plr != float('inf'):
            scene_plr[scene_id].append(plr)
    
    # Update per_scene with PLR
    for scene_data in data.get('per_scene', []):
        scene_id = scene_data.get('scene_id')
        if scene_id in scene_plr:
            plr_values = scene_plr[scene_id]
            scene_data['plr'] = float(np.mean(plr_values))
            scene_data['plr_std'] = float(np.std(plr_values))
    
    # Print summary
    print("\n" + "=" * 60)
    print("UPDATED PER-SCENE RESULTS (with PLR)")
    print("=" * 60)
    print(f"{'Scene ID':<20} {'FGE':>8} {'CR':>8} {'PLR':>8} {'Curv':>8} {'N':>6}")
    print("-" * 60)
    
    for scene in data.get('per_scene', []):
        scene_id = scene.get('scene_id', 'unknown')
        fge = scene.get('fge', 0)
        cr = scene.get('cr', 0)
        plr = scene.get('plr', 0)
        curv = scene.get('curv', 0)
        n = scene.get('num_episodes', 0)
        print(f"{scene_id:<20} {fge:>8.4f} {cr*100:>7.1f}% {plr:>8.4f} {curv:>8.4f} {n:>6}")
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("CONFIG & OVERALL SUMMARY")
    print("=" * 60)
    
    config = data.get('config', {})
    print(f"Method: {config.get('method', config.get('description', 'VLM+Planner'))}")
    print(f"Waypoints: {config.get('num_waypoints', 'N/A (uses PathPlanner with 8 ctrl points)')}")
    print(f"Dataset: {config.get('dataset', 'N/A')}")
    
    overall = data.get('overall', {})
    print(f"\nOverall Metrics:")
    print(f"  FGE:  {overall.get('fge_mean', 0):.4f} ± {overall.get('fge_std', 0):.4f}")
    print(f"  CR:   {overall.get('cr_mean', 0)*100:.2f}%")
    print(f"  PLR:  {overall.get('plr_mean', 0):.4f} ± {overall.get('plr_std', 0):.4f}")
    print(f"  Curv: {overall.get('curv_mean', 0):.4f} ± {overall.get('curv_std', 0):.4f}")
    
    # Save updated results
    output_path = results_path.parent / f"{results_path.stem}_fixed.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Fixed results saved to: {output_path}")
    
    return data


def compare_results(path1: str, path2: str):
    """Compare two results files"""
    
    print("\n" + "=" * 80)
    print("COMPARISON: VLM+Planner vs VLM Direct Trajectory")
    print("=" * 80)
    
    with open(path1, 'r') as f:
        data1 = json.load(f)
    with open(path2, 'r') as f:
        data2 = json.load(f)
    
    o1 = data1.get('overall', {})
    o2 = data2.get('overall', {})
    c1 = data1.get('config', {})
    c2 = data2.get('config', {})
    
    print(f"\n{'Metric':<25} {'VLM+Planner':>20} {'VLM Trajectory':>20} {'Winner':>12}")
    print("-" * 80)
    
    metrics = [
        ('FGE (↓ better)', 'fge_mean', 'lower'),
        ('CR % (↓ better)', 'cr_mean', 'lower'),
        ('PLR (→1 better)', 'plr_mean', 'closer_to_1'),
        ('Curvature (↓ better)', 'curv_mean', 'lower'),
    ]
    
    for name, key, better in metrics:
        v1 = o1.get(key, 0)
        v2 = o2.get(key, 0)
        
        if key == 'cr_mean':
            s1, s2 = f"{v1*100:.2f}%", f"{v2*100:.2f}%"
        else:
            s1, s2 = f"{v1:.4f}", f"{v2:.4f}"
        
        if better == 'lower':
            winner = "Planner" if v1 < v2 else "Trajectory"
        elif better == 'closer_to_1':
            winner = "Planner" if abs(v1 - 1) < abs(v2 - 1) else "Trajectory"
        else:
            winner = "Planner" if v1 > v2 else "Trajectory"
        
        print(f"{name:<25} {s1:>20} {s2:>20} {winner:>12}")
    
    print("\n" + "-" * 80)
    print(f"{'Waypoints':<25} {'8 (Bezier ctrl pts)':>20} {c2.get('num_waypoints', 16):>20}")
    print(f"{'Valid Episodes':<25} {o1.get('num_valid', 0):>20} {o2.get('num_valid', 0):>20}")
    print(f"{'Errors':<25} {o1.get('num_errors', 0):>20} {o2.get('num_errors', 0):>20}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: fix both and compare
        base_dir = Path(__file__).parent
        
        planner_path = base_dir / "vlm_planner_full/results.json"
        traj_path = base_dir / "vlm_trajectory_results/results.json"
        
        if planner_path.exists():
            fix_results(str(planner_path))
        
        if traj_path.exists():
            fix_results(str(traj_path))
        
        if planner_path.exists() and traj_path.exists():
            compare_results(str(planner_path), str(traj_path))
    else:
        for path in sys.argv[1:]:
            fix_results(path)
