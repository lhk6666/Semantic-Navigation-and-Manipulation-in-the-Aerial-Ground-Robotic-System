"""
VLM Trajectory Planner - Direct Trajectory Generation (No Goal Input)

This planner uses VLM to directly generate a feasible trajectory.
The VLM must:
1. Understand the navigation instruction to identify the target
2. Generate a collision-free trajectory from start to the identified target

Input: Image + Instruction + Start position (NO goal position)
Output: Target identification + Trajectory waypoints

This serves as a baseline to compare against:
1. VLM + Traditional Planner (semantic_map_planner.py + path_planner.py)
2. End-to-end VLA models
"""

import os
import cv2
import numpy as np
import base64
import json
import time
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part


class VLMTrajectoryPlanner:
    """
    VLM-based trajectory planner that directly generates waypoints.
    
    Given a semantic map image and navigation instruction, the VLM:
    1. Identifies the target from the instruction
    2. Outputs a sequence of waypoints forming a collision-free path
    
    NO goal position is provided - VLM must understand the instruction.
    """
    
    def __init__(self, num_waypoints: int = 16):
        """
        Args:
            num_waypoints: Number of waypoints to generate (default 16 to match VLA)
        """
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        vertexai.init(
            project=cfg["project_id"],
        )
        
        self.num_waypoints = num_waypoints
        
        # Use flash model for trajectory generation
        self.model = GenerativeModel(
            cfg["flash_model"],
            system_instruction=self._get_system_prompt()
        )
        
        self.generation_config = {
            "max_output_tokens": 8192,  # Increased for full trajectory JSON
            "temperature": 0.0,  # Lower temperature for more consistent paths
            "top_p": 0.0,
            "response_mime_type": "application/json",  # Force JSON output
        }
        
        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]
    
    def _get_system_prompt(self) -> str:
        return f"""You are a robot trajectory planner analyzing TOP-DOWN VIEW (bird's eye view) semantic maps.

The image shows a 2D floor plan from above with:
- Different colored regions representing furniture and objects (tables, chairs, sofas, beds, etc.)
- Dark/black regions are typically obstacles or walls
- Lighter regions are typically navigable floor space
- A GREEN dot marks the robot's current START position

Your task is to:
1. UNDERSTAND the navigation instruction to identify the TARGET object/location
2. GENERATE a smooth, collision-free trajectory from START to TARGET

COORDINATE SYSTEM:
- (0, 0) is the TOP-LEFT corner
- (1, 1) is the BOTTOM-RIGHT corner
- x increases LEFT to RIGHT
- y increases TOP to BOTTOM

INSTRUCTION UNDERSTANDING:
- "left/right" refers to x-axis direction (left = smaller x, right = larger x)
- "upper/top" means smaller y values, "lower/bottom" means larger y values
- Pay attention to relative positions (e.g., "the table on the left" vs "the table on the right")
- Identify the specific target object mentioned in the instruction

OUTPUT REQUIREMENTS:
1. First identify the TARGET from the instruction
2. Generate exactly {self.num_waypoints} waypoints
3. First waypoint should be near the START position (green marker)
4. Last waypoint should be at the TARGET location
5. Waypoints should form a smooth path avoiding obstacles
6. Keep reasonable spacing between consecutive waypoints
7. Stay away from dark/obstacle regions

Output ONLY a JSON object in this exact format:
{{
    "target": {{"name": "identified target object", "x": 0.7, "y": 0.5}},
    "trajectory": [
        {{"x": 0.15, "y": 0.25}},
        {{"x": 0.18, "y": 0.30}},
        ... (exactly {self.num_waypoints} points total)
    ],
    "reasoning": "Brief explanation of target identification and path planning"
}}

PLANNING STRATEGY:
1. Read the instruction carefully to identify the target object
2. Locate the target in the semantic map based on color/shape/position
3. Find the start position (green marker)
4. Identify obstacles (dark regions, furniture to avoid)
5. Plan a smooth path from start to target, curving around obstacles"""

    def draw_markers(self, image: np.ndarray, start: np.ndarray,
                     grid_size: int = 5) -> np.ndarray:
        """
        Draw start marker and optional grid on image
        NOTE: No goal marker - VLM must identify target from instruction
        
        Args:
            image: Input image (H, W, 3) BGR format
            start: Start position in normalized coords [0,1]
            grid_size: Grid divisions (0 to disable)
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # Draw grid if requested
        if grid_size > 0:
            for i in range(1, grid_size):
                x = int(w * i / grid_size)
                cv2.line(img, (x, 0), (x, h), (128, 128, 128), 1)
                y = int(h * i / grid_size)
                cv2.line(img, (0, y), (w, y), (128, 128, 128), 1)
        
        # Draw start marker (green circle) - VLM needs to know where robot is
        start_px = (int(start[0] * w), int(start[1] * h))
        cv2.circle(img, start_px, 8, (0, 255, 0), -1)  # Filled green
        cv2.circle(img, start_px, 8, (0, 0, 0), 2)      # Black border
        cv2.putText(img, "S", (start_px[0]-5, start_px[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # NO goal marker - VLM must identify target from instruction
        
        return img
    
    def prepare_image(self, image: np.ndarray, start: np.ndarray,
                      add_grid: bool = True) -> str:
        """
        Prepare image for VLM with start marker only
        
        Args:
            image: Input image (H, W, 3) RGB format
            start: Start position [x, y] normalized [0,1]
            add_grid: Whether to add coordinate grid
            
        Returns:
            base64 encoded JPEG string
        """
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize to reasonable size
        target_size = 512
        h, w = img_bgr.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Add start marker only (no goal marker)
        grid_size = 5 if add_grid else 0
        img_marked = self.draw_markers(img_resized, start, grid_size)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpeg', img_marked, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode()
    
    def generate_trajectory(self, image: np.ndarray, instruction: str,
                           start: np.ndarray,
                           add_grid: bool = True) -> Dict[str, Any]:
        """
        Generate a trajectory from start to target (VLM identifies target from instruction)
        
        Args:
            image: Input image (H, W, 3) RGB format
            instruction: Navigation instruction (VLM uses this to find target)
            start: Start position [x, y] normalized [0,1]
            add_grid: Whether to add coordinate grid overlay
            
        Returns:
            Dictionary with:
                - target: Identified target {name, x, y}
                - trajectory: List of [x, y] waypoints
                - reasoning: VLM's explanation
                - error: Error message if failed
        """
        img_base64 = self.prepare_image(image, start, add_grid)
        
        contents = [
            Part.from_data(
                data=base64.b64decode(img_base64),
                mime_type="image/jpeg"
            ),
            Part.from_text(f"""This is a top-down semantic map of an indoor environment.

The GREEN dot marks the robot's current START position at approximately ({start[0]:.2f}, {start[1]:.2f}).

Navigation instruction: "{instruction}"

Your task:
1. Understand the instruction to identify the TARGET object/location in the image
2. Generate a trajectory of exactly {self.num_waypoints} waypoints from START to TARGET

The path should:
1. Start near the green marker (robot's position)
2. End at the TARGET location (you must identify from instruction)
3. Avoid all obstacles (dark regions, furniture)
4. Be smooth and efficient

Output JSON with target info and trajectory array.""")
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                result = self._parse_response(response.text, start)
                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    # Return error - no fallback since we don't know the goal
                    return {
                        "target": None,
                        "trajectory": [],
                        "reasoning": "",
                        "error": str(e)
                    }
    
    def _parse_response(self, response_text: str, start: np.ndarray) -> Dict[str, Any]:
        """Parse VLM response to extract target and trajectory"""
        result = {
            "target": None,
            "trajectory": [],
            "reasoning": "",
            "raw_response": response_text,
            "error": None
        }
        
        try:
            text = response_text.strip()
            
            # Extract JSON from various formats
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]
            
            text = text.strip()
            
            # Try to fix truncated JSON
            if not text.endswith("}"):
                # Find last complete trajectory point
                last_brace = text.rfind("}")
                if last_brace > 0:
                    # Try to close the JSON properly
                    text = text[:last_brace+1]
                    # Count open brackets and close them
                    open_brackets = text.count("[") - text.count("]")
                    open_braces = text.count("{") - text.count("}")
                    text += "]" * open_brackets + "}" * open_braces
            
            data = json.loads(text)
            
            # Extract target (VLM identified)
            if "target" in data and data["target"]:
                target = data["target"]
                if target.get("x") is not None and target.get("y") is not None:
                    result["target"] = {
                        "name": target.get("name", "unknown"),
                        "x": float(np.clip(target["x"], 0, 1)),
                        "y": float(np.clip(target["y"], 0, 1))
                    }
            
            # Extract trajectory
            if "trajectory" in data and data["trajectory"]:
                for pt in data["trajectory"]:
                    if pt.get("x") is not None and pt.get("y") is not None:
                        x = float(np.clip(pt["x"], 0, 1))
                        y = float(np.clip(pt["y"], 0, 1))
                        result["trajectory"].append([x, y])
            
            # Extract reasoning
            result["reasoning"] = data.get("reasoning", "")
            
            # Validate and adjust trajectory length
            if len(result["trajectory"]) > 0:
                # Get goal from trajectory end or target
                if result["target"]:
                    goal = np.array([result["target"]["x"], result["target"]["y"]])
                else:
                    goal = np.array(result["trajectory"][-1])
                
                if len(result["trajectory"]) < self.num_waypoints:
                    result["trajectory"] = self._interpolate_trajectory(
                        result["trajectory"], start, goal
                    )
                elif len(result["trajectory"]) > self.num_waypoints:
                    result["trajectory"] = self._subsample_trajectory(
                        result["trajectory"], self.num_waypoints
                    )
                
        except json.JSONDecodeError as e:
            result["error"] = f"JSON parse error: {e}"
        except Exception as e:
            result["error"] = f"Parse error: {e}"
            
        return result
    
    def _interpolate_trajectory(self, trajectory: List[List[float]], 
                                start: np.ndarray, goal: np.ndarray) -> List[List[float]]:
        """Interpolate trajectory to have exactly num_waypoints points"""
        if len(trajectory) == 0:
            return self._make_straight_line(start, goal)
        
        # Ensure start and end
        traj = np.array(trajectory)
        if len(traj) == 1:
            traj = np.vstack([start, traj, goal])
        
        # Calculate cumulative distances
        dists = np.zeros(len(traj))
        for i in range(1, len(traj)):
            dists[i] = dists[i-1] + np.linalg.norm(traj[i] - traj[i-1])
        
        if dists[-1] < 1e-6:
            return self._make_straight_line(start, goal)
        
        # Normalize distances
        dists = dists / dists[-1]
        
        # Interpolate at uniform intervals
        new_dists = np.linspace(0, 1, self.num_waypoints)
        new_traj = []
        for d in new_dists:
            idx = np.searchsorted(dists, d)
            if idx == 0:
                new_traj.append(traj[0].tolist())
            elif idx >= len(traj):
                new_traj.append(traj[-1].tolist())
            else:
                # Linear interpolation
                t = (d - dists[idx-1]) / (dists[idx] - dists[idx-1] + 1e-6)
                pt = traj[idx-1] + t * (traj[idx] - traj[idx-1])
                new_traj.append(pt.tolist())
        
        return new_traj
    
    def _subsample_trajectory(self, trajectory: List[List[float]], 
                              n: int) -> List[List[float]]:
        """Subsample trajectory to n points"""
        traj = np.array(trajectory)
        indices = np.linspace(0, len(traj)-1, n, dtype=int)
        return traj[indices].tolist()
    
    def _make_straight_line(self, start: np.ndarray, goal: np.ndarray) -> List[List[float]]:
        """Create a straight line trajectory as fallback"""
        start = np.array(start)
        goal = np.array(goal)
        trajectory = []
        for i in range(self.num_waypoints):
            t = i / (self.num_waypoints - 1)
            pt = start + t * (goal - start)
            trajectory.append(pt.tolist())
        return trajectory


if __name__ == "__main__":
    import zarr
    
    # Test with dataset
    dataset_path = '/media/dragon_llm/linux_ssd/vla_dp_224/val'
    
    with open(f'{dataset_path}/episode_meta.json', 'r') as f:
        meta = json.load(f)
    
    z = zarr.open_group(dataset_path)
    sample_ids = [m['sample_id'] for m in meta]
    unique_sample_ids = sorted(set(sample_ids))
    sample_id_to_idx = {sid: i for i, sid in enumerate(unique_sample_ids)}
    
    # Get first episode
    ep = meta[0]
    print(f'Episode 0:')
    print(f'  Instruction: {ep["instruction"]}')
    print(f'  GT Goal: {ep["goal"]}')
    print(f'  GT Start: {ep["start"]}')
    
    # Get image
    img_idx = sample_id_to_idx[ep['sample_id']]
    image = np.array(z['data/img'][img_idx])
    
    # Initialize planner
    planner = VLMTrajectoryPlanner(num_waypoints=16)
    
    # Generate trajectory (only provide start + instruction, NO goal)
    print('\nGenerating trajectory (VLM must identify target from instruction)...')
    start = np.array(ep['start'])
    
    result = planner.generate_trajectory(
        image=image,
        instruction=ep['instruction'],
        start=start
    )
    
    print(f'\nResult:')
    print(f'  VLM Identified Target: {result.get("target", "N/A")}')
    print(f'  Trajectory length: {len(result["trajectory"])}')
    if len(result["trajectory"]) > 0:
        print(f'  First point: {result["trajectory"][0]}')
        print(f'  Last point: {result["trajectory"][-1]}')
    print(f'  GT Start: {ep["start"]}')
    print(f'  GT Goal: {ep["goal"]}')
    print(f'  Reasoning: {result.get("reasoning", "N/A")}')
    if result.get("error"):
        print(f'  Error: {result["error"]}')
    
    # Calculate metrics
    gt_goal = np.array(ep["goal"])
    
    if len(result["trajectory"]) > 0:
        pred_final = np.array(result["trajectory"][-1])
        fge = np.linalg.norm(pred_final - gt_goal)
        print(f'\n  Final Goal Error (FGE): {fge:.4f}')
    
    if result.get("target"):
        pred_target = np.array([result["target"]["x"], result["target"]["y"]])
        target_error = np.linalg.norm(pred_target - gt_goal)
        print(f'  Target Detection Error: {target_error:.4f}')
