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
from typing import Optional, Tuple, Callable
import re

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, SafetySetting, Part
except Exception:  # pragma: no cover - backend dependency is optional
    vertexai = None
    GenerativeModel = None
    SafetySetting = None
    Part = None


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
    
    def __init__(
        self,
        use_finetuned: bool = False,
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        api_key_env: Optional[str] = None,
        before_request: Optional[Callable[[], None]] = None,
        http_timeout_ms: int = 120000,
    ):
        """
        Args:
            use_finetuned: Whether to use finetuned model (requires endpoint)
        """
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        self.cfg = cfg or {}
        self.before_request = before_request
        self.http_timeout_ms = int(http_timeout_ms)
        if self.http_timeout_ms <= 0:
            raise ValueError("http_timeout_ms must be positive")
        self.llm_backend = str(backend or self.cfg.get("llm_backend") or "vertexai").lower()
        if self.llm_backend not in {"vertexai", "genai"}:
            raise RuntimeError(f"Unsupported llm_backend={self.llm_backend!r}")
        self.genai_client = None
        self.api_key_env_name = None

        if self.llm_backend == "genai":
            self.api_key_env_name = str(
                api_key_env or self.cfg.get("genai_api_key_env") or "GEMINI_API_KEY"
            )
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env_name):
                raise RuntimeError(
                    "Invalid API-key environment variable name; do not provide a key value"
                )
            api_key = os.environ.get(self.api_key_env_name)
            if not api_key:
                raise RuntimeError(
                    "Missing Google AI Studio API key in environment variable "
                    f"{self.api_key_env_name}"
                )
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore
            except Exception as error:
                raise RuntimeError(
                    "google-genai is not installed; install it or select vertexai"
                ) from error
            self.model_name = str(
                model or self.cfg.get("genai_model") or "gemini-2.5-flash"
            )
            http_options = types.HttpOptions(
                timeout=self.http_timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self.genai_client = genai.Client(
                api_key=api_key,
                http_options=http_options,
            )
            self.model = None
        else:
            if vertexai is None or GenerativeModel is None:
                raise RuntimeError(
                    "vertexai SDK is not available; install it or select genai"
                )
            vertexai.init(
                project=self.cfg["project_id"],
                location=self.cfg["location"]
            )
            self.model_name = str(
                model or self.cfg.get("flash_model") or "gemini-2.5-flash"
            )
            self.model = GenerativeModel(
                self.model_name,
                system_instruction=self._get_system_prompt()
            )

        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        
        self.safety_settings = [] if self.llm_backend == "genai" else [
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

    def _safe_error(self, error: Exception) -> str:
        """Format an SDK error without ever exposing the configured API key."""
        message = str(error)
        if self.api_key_env_name:
            secret = os.environ.get(self.api_key_env_name)
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

    def _call_llm(self, image_bytes: bytes, prompt_text: str) -> str:
        """Call the selected backend with identical prompt/model parameters."""
        if self.before_request is not None:
            self.before_request()
        if self.llm_backend == "genai":
            from google.genai import types  # type: ignore

            if hasattr(types.Part, "from_bytes"):
                image_part = types.Part.from_bytes(
                    data=image_bytes, mime_type="image/jpeg"
                )
            else:  # pragma: no cover - old SDK compatibility
                image_part = types.Part(
                    inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)
                )
            if hasattr(types.Part, "from_text"):
                text_part = types.Part.from_text(text=prompt_text)
            else:  # pragma: no cover
                text_part = types.Part(text=prompt_text)
            config_kwargs = {
                "max_output_tokens": 8192,
                "temperature": 0.0,
                "top_p": 1.0,
                "response_mime_type": "application/json",
                "system_instruction": self._get_system_prompt(),
            }
            try:
                generate_config = types.GenerateContentConfig(**config_kwargs)
            except TypeError:  # pragma: no cover - old SDK compatibility
                config_kwargs.pop("response_mime_type", None)
                generate_config = types.GenerateContentConfig(**config_kwargs)
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[image_part, text_part])],
                config=generate_config,
            )
            return getattr(response, "text", None) or str(response)

        response = self.model.generate_content(
            contents=[
                Part.from_data(data=image_bytes, mime_type="image/jpeg"),
                Part.from_text(text=prompt_text),
            ],
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )
        return response.text
    
    def _get_system_prompt(self) -> str:
                return """You are analyzing a TOP-DOWN VIEW (bird's eye view) RGB image of an indoor environment for robot navigation.

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
1. The TARGET OBJECT (the object the instruction refers to) with its center and bounding box.
2. Other obstacle objects that might block a path, with their centers and bounding boxes.
3. A suggested START position if visible.
4. If multiple candidates exist, choose the one you judge most consistent with the instruction and overall scene layout.

Output ONLY valid JSON. Required keys are exactly:
{
    "target": {
        "name": "object_name",
        "center": [0.50, 0.70],
        "bbox": [0.40, 0.60, 0.60, 0.85],
        "confidence": "high/medium/low"
    },
    "obstacles": [
        {"name": "obstacle1", "center": [0.30, 0.40], "bbox": [0.25, 0.32, 0.38, 0.48]}
    ],
    "start": {"x": 0.20, "y": 0.30}
}

Important:
- All coordinates must be between 0.0 and 1.0
- bbox format is [xmin, ymin, xmax, ymax] in normalized coords
- Ensure xmin < xmax and ymin < ymax
- If uncertain, still provide your best guess and set confidence to "low""" 

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
        add_grid: bool = False,
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
        
        base_text = f"""This is a top-down RGB image of an indoor environment.

Navigation instruction: "{instruction}"

Please analyze the image and identify:
1. The target location based on the instruction
2. Any obstacles in the scene
3. A reasonable start position (if not specified, estimate from context)

Important: If the instruction asks for a SIDE of an object (left/right/upper/lower side), then the object itself must be treated as an obstacle.

CRITICAL: You MUST output a non-null "target" with numeric center [x,y] and
bbox [xmin,ymin,xmax,ymax] values in [0,1]. Each obstacle must also include a bbox.
If uncertain, still provide your best guess and set confidence to "low".

Output your analysis as JSON only."""

        image_bytes = base64.b64decode(img_base64)
        prompt_text = base_text

        # Keep this planner lightweight: minimal retries.
        max_retries = 2 if require_target else 1
        last_result = None
        for attempt in range(max_retries):
            try:
                response_text = self._call_llm(image_bytes, prompt_text)
                result = self._parse_response(response_text, instruction=instruction)
                last_result = result

                if require_target and (not result.get("target") or result.get("error")):
                    raise ValueError(f"Invalid/partial parse: {result.get('error', 'missing target')}")

                return result
                
            except Exception as e:
                safe_error = self._safe_error(e)
                print(f"Attempt {attempt + 1} failed: {safe_error}")
                if attempt < max_retries - 1:
                    # Simple retry note.
                    prompt_text = (
                        base_text
                        + "\n\nRetry: Return ONLY JSON. Include target.center as [x,y], "
                        "target.bbox as [xmin,ymin,xmax,ymax], and each obstacle.bbox "
                        "in the same format, all numeric and in [0,1]."
                    )
                else:
                    fallback = last_result if isinstance(last_result, dict) else {
                        "target": None,
                        "goal": None,
                        "approach_side": None,
                        "obstacles": [],
                        "start": None,
                        "raw_response": "",
                    }
                    fallback["error"] = safe_error
                    return fallback
    
    def _parse_response(self, response_text: str, instruction: str = "") -> dict:
        """Parse VLM response to extract structured data"""
        result = {
            "target": None,
            "goal": None,
            "approach_side": None,
            "obstacles": [],
            "start": None,
            "raw_response": response_text
        }

        def _clip01(v: float) -> float:
            return float(np.clip(float(v), 0.0, 1.0))

        def _parse_center(obj: dict) -> Optional[Tuple[float, float]]:
            if not isinstance(obj, dict):
                return None
            c = obj.get("center")
            if isinstance(c, (list, tuple)) and len(c) == 2:
                return (_clip01(c[0]), _clip01(c[1]))
            if obj.get("x") is not None and obj.get("y") is not None:
                return (_clip01(obj["x"]), _clip01(obj["y"]))
            return None

        def _parse_bbox(obj: dict) -> Optional[Tuple[float, float, float, float]]:
            if not isinstance(obj, dict):
                return None
            b = obj.get("bbox")
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                return None
            xmin, ymin, xmax, ymax = (_clip01(b[0]), _clip01(b[1]), _clip01(b[2]), _clip01(b[3]))
            if xmax <= xmin or ymax <= ymin:
                return None
            return (xmin, ymin, xmax, ymax)

        def _side_from_dir(dx: float, dy: float) -> str:
            if dx < 0:
                return "left"
            if dx > 0:
                return "right"
            if dy < 0:
                return "top"
            if dy > 0:
                return "bottom"
            return "none"

        def _goal_from_bbox_and_side(bbox: Tuple[float, float, float, float], side: str, center: Tuple[float, float]) -> Tuple[float, float]:
            xmin, ymin, xmax, ymax = bbox
            cx, cy = center
            offset = 0.02
            if side == "left":
                return (_clip01(xmin - offset), cy)
            if side == "right":
                return (_clip01(xmax + offset), cy)
            if side == "top":
                return (cx, _clip01(ymin - offset))
            if side == "bottom":
                return (cx, _clip01(ymax + offset))
            return (cx, cy)
        
        try:
            text = self._extract_json_text(response_text)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                repaired = self._repair_json_text(text)
                data = json.loads(repaired)
            
            # Extract target object
            if "target" in data and data["target"]:
                tgt = data["target"]
                center = _parse_center(tgt)
                bbox = _parse_bbox(tgt)
                if center is not None and bbox is not None:
                    result["target"] = {
                        "name": str(tgt.get("name", "unknown")),
                        "center": [float(center[0]), float(center[1])],
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "confidence": str(tgt.get("confidence", "medium")),
                    }
            
            # Extract obstacles (as objects with bboxes)
            if "obstacles" in data and data["obstacles"]:
                for obs in data["obstacles"]:
                    center = _parse_center(obs)
                    bbox = _parse_bbox(obs)
                    if center is None or bbox is None:
                        continue
                    result["obstacles"].append({
                        "name": str(obs.get("name", "obstacle")),
                        "center": [float(center[0]), float(center[1])],
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    })
            
            # Extract start
            if "start" in data and data["start"]:
                start = data["start"]
                if start.get("x") is not None and start.get("y") is not None:
                    result["start"] = {
                        "x": _clip01(start["x"]),
                        "y": _clip01(start["y"])
                    }

            # Derive approach side from instruction and compute the navigation goal.
            side_dir = self._infer_side_direction(instruction)
            if result.get("target") is not None:
                bbox = tuple(result["target"]["bbox"])  # type: ignore[assignment]
                center = tuple(result["target"]["center"])  # type: ignore[assignment]
                if side_dir is None:
                    result["approach_side"] = "none"
                    gx, gy = center
                else:
                    dx, dy = side_dir
                    result["approach_side"] = _side_from_dir(dx, dy)
                    gx, gy = _goal_from_bbox_and_side(bbox=bbox, side=result["approach_side"], center=center)
                result["goal"] = {"x": float(gx), "y": float(gy)}
            
            # Strict mode: if no target parsed, report an error.
            if result.get("target") is None:
                result["error"] = "Missing/invalid 'target' (need center + bbox)"
                    
        except json.JSONDecodeError as e:
            result["error"] = f"JSON parse error: {e}"
        except Exception as e:
            result["error"] = f"Parse error: {e}"
            
        return result


if __name__ == "__main__":
    import zarr
    
    # Test with dataset
    dataset_path = '/media/dragon_llm/linux_ssd/vla_dataset_unified_static_v13/val'
    
    with open(f'{dataset_path}/episode_meta.json', 'r') as f:
        meta = json.load(f)
    
    dataset_root = Path(dataset_path)
    zarr_root = dataset_root / 'dataset.zarr' if (dataset_root / 'dataset.zarr').exists() else dataset_root
    z = zarr.open_group(str(zarr_root), mode='r')

    # Prefer explicit mapping if present (unified format)
    sample_indices_path = dataset_root / 'sample_indices.json'
    if not sample_indices_path.exists():
        sample_indices_path = zarr_root.parent / 'sample_indices.json'
    if sample_indices_path.exists():
        with open(sample_indices_path, 'r') as f:
            sample_id_to_idx = json.load(f)
    else:
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
    planner = SemanticMapPlanner()
    
    # Analyze scene
    print('\nAnalyzing scene...')
    result = planner.analyze_scene(image, ep['instruction'])
    
    print(f'\nVLM Result:')
    print(f'  Target: {result["target"]}')
    print(f'  Obstacles: {result["obstacles"]}')
    print(f'  Start: {result["start"]}')
    
    if result["target"]:
        pred_goal = result["target"]["center"]
        gt_goal = ep["goal"]
        error = np.linalg.norm(np.array(pred_goal) - np.array(gt_goal))
        print(f'\n  Predicted goal: {pred_goal}')
        print(f'  GT goal: {gt_goal}')
        print(f'  Error: {error:.4f}')
