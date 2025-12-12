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

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part


class SemanticMapPlanner:
    """Planner for semantic top-down maps (Matterport-style)"""
    
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
            "temperature": 0.3,
            "top_p": 0.8,
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
1. The TARGET object mentioned in the instruction and its approximate center position
2. Any OBSTACLES that might block a direct path
3. A suggested START position if visible (usually the robot's current location)

Output ONLY valid JSON in this exact format:
{
    "target": {"name": "object_name", "x": 0.5, "y": 0.7, "confidence": "high/medium/low"},
    "obstacles": [{"name": "obstacle1", "x": 0.3, "y": 0.4}],
    "start": {"x": 0.2, "y": 0.3}
}

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
    
    def analyze_scene(self, image: np.ndarray, instruction: str, 
                     add_grid: bool = True) -> dict:
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
        
        contents = [
            Part.from_data(
                data=base64.b64decode(img_base64),
                mime_type="image/jpeg"
            ),
            Part.from_text(f"""This is a top-down semantic map of an indoor environment.

Navigation instruction: "{instruction}"

Please analyze the image and identify:
1. The target location based on the instruction
2. Any obstacles in the scene
3. A reasonable start position (if not specified, estimate from context)

Output your analysis as JSON only.""")
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                # Parse response
                result = self._parse_response(response.text)
                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return {"target": None, "obstacles": [], "start": None, "error": str(e)}
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse VLM response to extract structured data"""
        result = {
            "target": None,
            "obstacles": [],
            "start": None,
            "raw_response": response_text
        }
        
        try:
            # Clean response
            text = response_text.strip()
            
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            text = text.strip()
            data = json.loads(text)
            
            # Extract target
            if "target" in data and data["target"]:
                target = data["target"]
                if target.get("x") is not None and target.get("y") is not None:
                    result["target"] = {
                        "name": target.get("name", "unknown"),
                        "x": float(target["x"]),
                        "y": float(target["y"]),
                        "confidence": target.get("confidence", "medium")
                    }
            
            # Extract obstacles
            if "obstacles" in data and data["obstacles"]:
                for obs in data["obstacles"]:
                    if obs.get("x") is not None and obs.get("y") is not None:
                        result["obstacles"].append({
                            "name": obs.get("name", "obstacle"),
                            "x": float(obs["x"]),
                            "y": float(obs["y"])
                        })
            
            # Extract start
            if "start" in data and data["start"]:
                start = data["start"]
                if start.get("x") is not None and start.get("y") is not None:
                    result["start"] = {
                        "x": float(start["x"]),
                        "y": float(start["y"])
                    }
                    
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
