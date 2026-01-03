"""
Semantic Map Planner for VLA Dataset Evaluation

This planner is designed to work with Matterport-style top-down semantic maps,
not real camera images. It uses Gemini to understand the scene and identify
target objects from a bird's eye view.
"""

import os
import cv2
import numpy as np
import base64
import json
import time
import yaml
from pathlib import Path
from typing import Optional, Tuple
import re

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part


class SemanticMapPlanner:
    """Planner for semantic top-down maps (Matterport-style)"""

    def _extract_json_text(self, response_text: str) -> str:
        text = (response_text or "").strip()

        # Prefer fenced JSON blocks.
        if "```json" in text:
            try:
                text = text.split("```json", 1)[1].split("```", 1)[0]
            except Exception:
                pass
        elif "```" in text:
            try:
                text = text.split("```", 1)[1].split("```", 1)[0]
            except Exception:
                pass

        text = text.strip()

        # Extract the largest {...} region.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return text.strip()

    def _repair_json_text(self, text: str) -> str:
        t = (text or "").strip()

        # Remove standalone commas on their own line.
        t = re.sub(r"\n\s*,\s*\n", "\n", t)

        # Fix a common Gemini mistake: missing closing brace for the 'target' object.
        # Example:
        #   "target": { ... "confidence": "high" ,\n    "obstacles": ...
        t = re.sub(
            r'("confidence"\s*:\s*"[^"]+")\s*,\s*(\s*"obstacles"\s*:)',
            r'\1\n    },\n\2',
            t,
        )

        # Remove trailing commas before closing braces/brackets.
        t = re.sub(r",\s*([}\]])", r"\1", t)

        # Truncate after the last closing brace and then balance braces/brackets.
        last_brace = t.rfind("}")
        if last_brace > 0:
            t = t[: last_brace + 1]

        open_brackets = t.count("[") - t.count("]")
        open_braces = t.count("{") - t.count("}")
        if open_brackets > 0:
            t += "]" * open_brackets
        if open_braces > 0:
            t += "}" * open_braces

        return t

    def _infer_side_direction(self, instruction: str) -> Optional[Tuple[float, float]]:
        """Infer if the instruction requests a directional side of an object.

        Returns:
            (dx, dy) in normalized image coords (x right+, y down+) if a side is requested,
            otherwise None.
        """
        if not instruction:
            return None
        t = instruction.lower()

        left_keys = [
            "left of", "to the left of", "on the left of", "left side", "left-hand side",
            "左边", "左侧",
        ]
        right_keys = [
            "right of", "to the right of", "on the right of", "right side", "right-hand side",
            "右边", "右侧",
        ]
        up_keys = [
            "above", "upper", "top of", "on top of", "upper side", "top side",
            "上方", "上面", "上侧",
        ]
        down_keys = [
            "below", "lower", "bottom of", "under", "bottom side", "lower side",
            "下方", "下面", "下侧",
        ]

        if any(k in t for k in left_keys):
            return (-1.0, 0.0)
        if any(k in t for k in right_keys):
            return (1.0, 0.0)
        if any(k in t for k in up_keys):
            return (0.0, -1.0)
        if any(k in t for k in down_keys):
            return (0.0, 1.0)
        return None
    
    def __init__(self, use_finetuned: bool = False):
        """
        Args:
            use_finetuned: Whether to use finetuned model (requires endpoint)
        """
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        vertexai.init(
            project=cfg["project_id"],
        )
        
        # Use flash model for semantic maps (more general)
        self.model = GenerativeModel(
            cfg["flash_model"],
            system_instruction=self._get_system_prompt()
        )
        
        self.generation_config = {
            "max_output_tokens": 4096,
            "temperature": 0.0,
            "top_p": 0.0,
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
        return """You are analyzing a TOP-DOWN VIEW (bird's eye view) semantic map of an indoor environment for robot navigation.

The image is a 2D floor plan view from above, showing:
- Different colored regions representing different areas/objects (tables, chairs, sofas, beds, etc.)
- The layout of the room/space
- Obstacles and furniture

Your task is to identify objects and their locations in NORMALIZED COORDINATES where:
- (0, 0) is the TOP-LEFT corner of the image
- (1, 1) is the BOTTOM-RIGHT corner of the image
- x increases from LEFT to RIGHT
- y increases from TOP to BOTTOM

Given a navigation instruction, identify:
1. The navigation GOAL location (what the robot should move to) in normalized coordinates.
2. Any OBSTACLES that might block a direct path.
3. A suggested START position if visible (usually the robot's current location).

Directional/side instructions (IMPORTANT):
- If the instruction says to go to a SIDE of an object (e.g., "left of the table", "right side of the sofa", "upper of the bed"), then:
    - Set "target" to the GOAL point on that requested side (slightly outside the object's region).
    - The referenced object itself MUST be treated as an obstacle (otherwise a planner may route through it).
    - Include the object's approximate center as "target_object" (optional field) AND also include it in "obstacles".

Output ONLY valid JSON. Required keys are exactly:
{
    "target": {"name": "object_name", "x": 0.5, "y": 0.7, "confidence": "high/medium/low"},
    "obstacles": [{"name": "obstacle1", "x": 0.3, "y": 0.4}],
    "start": {"x": 0.2, "y": 0.3}
}

If and only if the instruction is directional (go to a side of an object), you may add this optional key:
        "target_object": {"name": "object_name", "x": 0.5, "y": 0.7}

Important:
- Coordinates must be between 0.0 and 1.0
- If you cannot identify an object with confidence, set confidence to "low"
- Look for semantic regions that match the target description (e.g., "table" might be a rectangular region)
- Consider the relative positions mentioned in instructions (e.g., "lower table" means the table at higher y value)
- "left/right" refers to x-axis, "upper/lower/top/bottom" refers to y-axis"""

    def draw_grid(self, image: np.ndarray, grid_size: int = 5) -> np.ndarray:
        """Draw a coordinate grid overlay on the image"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Draw grid lines
        for i in range(1, grid_size):
            # Vertical lines
            x = int(w * i / grid_size)
            cv2.line(img, (x, 0), (x, h), (128, 128, 128), 1)
            # Horizontal lines
            y = int(h * i / grid_size)
            cv2.line(img, (0, y), (w, y), (128, 128, 128), 1)
        
        # Draw coordinate labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                x = int(w * i / grid_size)
                y = int(h * j / grid_size)
                label = f"({i/grid_size:.1f},{j/grid_size:.1f})"
                cv2.putText(img, label, (max(0, x-20), max(10, y)), 
                           font, font_scale, (255, 255, 255), 1)
        
        return img
    
    def prepare_image(self, image: np.ndarray, add_grid: bool = True) -> str:
        """
        Prepare image for VLM
        
        Args:
            image: Input image (H, W, 3) RGB format
            add_grid: Whether to add coordinate grid overlay
            
        Returns:
            base64 encoded JPEG string
        """
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize to reasonable size for VLM (not too large)
        target_size = 512
        h, w = img_bgr.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Add grid if requested
        if add_grid:
            img_resized = self.draw_grid(img_resized)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpeg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode()
    
    def analyze_scene(
        self,
        image: np.ndarray,
        instruction: str,
        add_grid: bool = True,
        require_target: bool = True,
    ) -> dict:
        """
        Analyze the scene and identify target/obstacles
        
        Args:
            image: Input image (H, W, 3) RGB format
            instruction: Navigation instruction
            add_grid: Whether to add coordinate grid
            
        Returns:
            Dictionary with target, obstacles, start positions
        """
        img_base64 = self.prepare_image(image, add_grid)
        
        base_text = f"""This is a top-down semantic map of an indoor environment.

Navigation instruction: "{instruction}"

Please analyze the image and identify:
1. The target location based on the instruction
2. Any obstacles in the scene
3. A reasonable start position (if not specified, estimate from context)

Important: If the instruction asks for a SIDE of an object (left/right/upper/lower side), then the object itself must be treated as an obstacle.

CRITICAL: You MUST output a non-null "target" with numeric "x" and "y" in [0, 1].
If uncertain, still provide your best guess and set confidence to "low".

Output your analysis as JSON only."""

        contents = [
            Part.from_data(
                data=base64.b64decode(img_base64),
                mime_type="image/jpeg"
            ),
            Part.from_text(base_text)
        ]

        max_retries = 5 if require_target else 3
        last_result = None
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                # Parse response
                result = self._parse_response(response.text, instruction=instruction)
                last_result = result

                if require_target and (not result.get("target") or result.get("error")):
                    raise ValueError(f"Invalid/partial parse: {result.get('error', 'missing target')}")

                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Escalate instructions on subsequent retries.
                    contents = [
                        contents[0],
                        Part.from_text(
                            base_text
                            + "\n\nRetry note: Your previous output was invalid or missing 'target'. "
                              "Return ONLY JSON and ensure target.x/target.y are present (numbers 0..1)."
                        ),
                    ]
                    time.sleep(1)
                else:
                    # Final fallback: always return a target so downstream code can proceed.
                    fallback = last_result if isinstance(last_result, dict) else {
                        "target": None,
                        "target_object": None,
                        "obstacles": [],
                        "start": None,
                        "raw_response": "",
                    }
                    if require_target and not fallback.get("target"):
                        fallback["target"] = {
                            "name": "unknown",
                            "x": 0.5,
                            "y": 0.5,
                            "confidence": "low",
                        }
                    fallback["error"] = str(e)
                    return fallback
    
    def _parse_response(self, response_text: str, instruction: str = "") -> dict:
        """Parse VLM response to extract structured data"""
        result = {
            "target": None,
            "target_object": None,
            "obstacles": [],
            "start": None,
            "raw_response": response_text
        }
        
        try:
            text = self._extract_json_text(response_text)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                repaired = self._repair_json_text(text)
                data = json.loads(repaired)
            
            # Extract target
            if "target" in data and data["target"]:
                target = data["target"]
                if target.get("x") is not None and target.get("y") is not None:
                    result["target"] = {
                        "name": target.get("name", "unknown"),
                        "x": float(np.clip(float(target["x"]), 0.0, 1.0)),
                        "y": float(np.clip(float(target["y"]), 0.0, 1.0)),
                        "confidence": target.get("confidence", "medium")
                    }

            # Optional: extract explicit target object center (for directional tasks)
            if "target_object" in data and data["target_object"]:
                tobj = data["target_object"]
                if tobj.get("x") is not None and tobj.get("y") is not None:
                    result["target_object"] = {
                        "name": tobj.get("name", (result["target"]["name"] if result["target"] else "unknown")),
                        "x": float(np.clip(float(tobj["x"]), 0.0, 1.0)),
                        "y": float(np.clip(float(tobj["y"]), 0.0, 1.0)),
                    }
            
            # Extract obstacles
            if "obstacles" in data and data["obstacles"]:
                for obs in data["obstacles"]:
                    if obs.get("x") is not None and obs.get("y") is not None:
                        result["obstacles"].append({
                            "name": obs.get("name", "obstacle"),
                            "x": float(np.clip(float(obs["x"]), 0.0, 1.0)),
                            "y": float(np.clip(float(obs["y"]), 0.0, 1.0))
                        })
            
            # Extract start
            if "start" in data and data["start"]:
                start = data["start"]
                if start.get("x") is not None and start.get("y") is not None:
                    result["start"] = {
                        "x": float(np.clip(float(start["x"]), 0.0, 1.0)),
                        "y": float(np.clip(float(start["y"]), 0.0, 1.0))
                    }

            # Post-process directional tasks: ensure the target object is treated as an obstacle.
            side_dir = self._infer_side_direction(instruction)
            if side_dir is not None and result.get("target") is not None:
                dx, dy = side_dir

                # If the model already included the target object as an obstacle with the same name, reuse it.
                tgt_name = str(result["target"].get("name", "unknown"))
                same_name_obs = None
                for obs in result.get("obstacles", []):
                    if str(obs.get("name", "")) == tgt_name:
                        same_name_obs = obs
                        break

                if result.get("target_object") is None and same_name_obs is not None:
                    result["target_object"] = {
                        "name": tgt_name,
                        "x": float(same_name_obs["x"]),
                        "y": float(same_name_obs["y"]),
                    }

                # If we still don't have an explicit object center, assume the current target is the object center
                # and shift the navigation goal to the requested side.
                if result.get("target_object") is None:
                    obj_cx = float(result["target"]["x"])
                    obj_cy = float(result["target"]["y"])
                    result["target_object"] = {"name": tgt_name, "x": obj_cx, "y": obj_cy}

                    offset = 0.06
                    goal_x = float(np.clip(obj_cx + dx * offset, 0.0, 1.0))
                    goal_y = float(np.clip(obj_cy + dy * offset, 0.0, 1.0))
                    result["target"]["x"] = goal_x
                    result["target"]["y"] = goal_y

                # Ensure the object center is present in obstacles.
                obj = result["target_object"]
                already = False
                for obs in result.get("obstacles", []):
                    if abs(float(obs.get("x", 0.0)) - float(obj["x"])) < 0.02 and abs(float(obs.get("y", 0.0)) - float(obj["y"])) < 0.02:
                        already = True
                        break
                if not already:
                    result["obstacles"].append({"name": obj.get("name", tgt_name), "x": float(obj["x"]), "y": float(obj["y"])})
                    
        except json.JSONDecodeError as e:
            result["error"] = f"JSON parse error: {e}"
        except Exception as e:
            result["error"] = f"Parse error: {e}"
            
        return result


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
    print(f'  GT Goal (normalized): {ep["goal"]}')
    print(f'  GT Start (normalized): {ep["start"]}')
    
    # Get image
    img_idx = sample_id_to_idx[ep['sample_id']]
    image = np.array(z['data/img'][img_idx])
    
    # Initialize planner
    planner = SemanticMapPlanner()
    
    # Analyze scene
    print('\nAnalyzing scene...')
    result = planner.analyze_scene(image, ep['instruction'])
    
    print(f'\nVLM Result:')
    print(f'  Target: {result["target"]}')
    print(f'  Obstacles: {result["obstacles"]}')
    print(f'  Start: {result["start"]}')
    
    if result["target"]:
        pred_goal = [result["target"]["x"], result["target"]["y"]]
        gt_goal = ep["goal"]
        error = np.linalg.norm(np.array(pred_goal) - np.array(gt_goal))
        print(f'\n  Predicted goal: {pred_goal}')
        print(f'  GT goal: {gt_goal}')
        print(f'  Error: {error:.4f}')
