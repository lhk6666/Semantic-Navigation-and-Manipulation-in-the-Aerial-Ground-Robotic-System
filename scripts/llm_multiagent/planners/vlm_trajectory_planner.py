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
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, SafetySetting, Part
except Exception:  # pragma: no cover
    vertexai = None
    GenerativeModel = None
    SafetySetting = None
    Part = None


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

        self.cfg = cfg or {}
        self.llm_backend = (self.cfg.get("llm_backend") or "vertexai").lower()
        
        self.num_waypoints = num_waypoints

        self.max_output_tokens = int(self.cfg.get("vlm_traj_max_output_tokens", 8192))
        self.allow_fallback = bool(self.cfg.get("vlm_traj_allow_fallback", True))

        self.genai_client = None
        self.genai_model = None

        if self.llm_backend == "genai":
            # Google GenAI (API key) backend (no Vertex AI)
            api_key_env = str(self.cfg.get("genai_api_key_env", "GOOGLE_CLOUD_API_KEY") or "GOOGLE_CLOUD_API_KEY")
            # Guardrail: don't let users paste the API key itself into YAML.
            # Env var names should look like: [A-Za-z_][A-Za-z0-9_]*
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", api_key_env):
                raise RuntimeError(
                    "Invalid genai_api_key_env in config.yaml. "
                    "It must be an environment variable NAME (e.g., GOOGLE_CLOUD_API_KEY), not the API key value."
                )

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key in environment variable {api_key_env}. "
                    f"Export it (e.g., export {api_key_env}=... ) or change genai_api_key_env in config.yaml."
                )
            try:
                from google import genai  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "google-genai is not installed. Install it (pip install google-genai) "
                    "or switch llm_backend back to vertexai."
                ) from e

            # Match the sample style (no Vertex AI)
            self.genai_client = genai.Client(api_key=api_key)
            self.genai_model = self.cfg.get("genai_model", "gemini-3-pro-preview")
        else:
            # Vertex AI backend
            if vertexai is None or GenerativeModel is None:
                raise RuntimeError(
                    "vertexai SDK is not available. Install vertexai or set llm_backend: genai."
                )

            vertexai.init(project=self.cfg["project_id"], location=self.cfg["location"])
        
        # Use flash model for trajectory generation
        self.model = None
        if self.llm_backend != "genai":
            self.model = GenerativeModel(
                self.cfg.get("flash_model"),
                system_instruction=self._get_system_prompt(),
            )
        
        self.generation_config = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": 0.0,  # Lower temperature for more consistent paths
            "top_p": 1.0,
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

    def _extract_json_text(self, response_text: str) -> str:
        text = (response_text or "").strip()

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
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return text.strip()

    def _repair_json_text(self, text: str) -> str:
        t = (text or "").strip()

        # Remove standalone commas on their own line.
        t = re.sub(r"\n\s*,\s*\n", "\n", t)

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

    def _looks_truncated(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        # Common truncation tails we saw: ends mid-number or mid-string
        if t.endswith(".") or t.endswith('"') or t.endswith("\\"):
            return True
        # If braces/brackets are imbalanced, it's likely truncated.
        if t.count("{") != t.count("}"):
            return True
        if t.count("[") != t.count("]"):
            return True
        return False

    def _infer_target_from_text(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of target even when JSON is truncated."""
        if not response_text:
            return None
        m = re.search(
            r'"target"\s*:\s*\{.*?"x"\s*:\s*([0-9]*\.?[0-9]+).*?"y"\s*:\s*([0-9]*\.?[0-9]+).*?\}',
            response_text,
            flags=re.S,
        )
        if not m:
            return None
        try:
            x = float(np.clip(float(m.group(1)), 0, 1))
            y = float(np.clip(float(m.group(2)), 0, 1))
            name_m = re.search(r'"target"\s*:\s*\{.*?"name"\s*:\s*"([^"]+)"', response_text, flags=re.S)
            name = name_m.group(1) if name_m else "unknown"
            return {"name": name, "x": round(x, 2), "y": round(y, 2)}
        except Exception:
            return None

    def _call_llm(self, image_bytes: bytes, prompt_text: str) -> str:
        """Call configured LLM backend and return response text."""
        if self.llm_backend == "genai":
            from google.genai import types  # type: ignore

            parts = []
            # Image part
            if hasattr(types.Part, "from_bytes"):
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            else:
                # Fallback for older google-genai versions
                parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)))

            # Text part
            if hasattr(types.Part, "from_text"):
                parts.append(types.Part.from_text(text=prompt_text))
            else:
                parts.append(types.Part(text=prompt_text))

            contents = [types.Content(role="user", parts=parts)]

            # Optional knobs to match the sample template
            use_search = bool(self.cfg.get("genai_use_google_search", False))
            thinking_level = self.cfg.get("genai_thinking_level", None)

            tools = None
            if use_search:
                # SDK casing varies; prefer googleSearch as in the sample.
                try:
                    tools = [types.Tool(googleSearch=types.GoogleSearch())]
                except Exception:
                    try:
                        tools = [types.Tool(google_search=types.GoogleSearch())]
                    except Exception:
                        tools = None

            cfg_kwargs = {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": self.max_output_tokens,
            }
            if tools is not None:
                cfg_kwargs["tools"] = tools

            if thinking_level:
                try:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=str(thinking_level))
                except Exception:
                    pass

            # Some SDK versions support forcing JSON mime type.
            try:
                cfg_kwargs["response_mime_type"] = "application/json"
                gen_cfg = types.GenerateContentConfig(**cfg_kwargs)
            except TypeError:
                cfg_kwargs.pop("response_mime_type", None)
                gen_cfg = types.GenerateContentConfig(**cfg_kwargs)

            resp = self.genai_client.models.generate_content(
                model=self.genai_model,
                contents=contents,
                config=gen_cfg,
            )
            return getattr(resp, "text", None) or str(resp)

        # Vertex AI
        contents = [
            Part.from_data(data=image_bytes, mime_type="image/jpeg"),
            Part.from_text(text=prompt_text),
        ]
        response = self.model.generate_content(
            contents=contents,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )
        return response.text
    
    def _get_system_prompt(self) -> str:
        return f"""You are a robot navigation policy operating on a TOP-DOWN (bird’s-eye-view) RGB map and a natural-language instruction.

INPUTS
- Image: a top-down RGB map (RGB-only; no explicit obstacle mask).
- Text: (1) Instruction and (2) START coordinate in normalized image coordinates.

START (AUTHORITATIVE)
- The robot START is provided in text.
- The image may also show a green dot, but if there is any ambiguity, trust the text START.

COORDINATE SYSTEM (NORMALIZED)
- (0.00, 0.00) is the TOP-LEFT corner of the image
- (1.00, 1.00) is the BOTTOM-RIGHT corner
- x increases left -> right; y increases top -> bottom
- All output coordinates MUST have exactly 2 digits after the decimal (e.g., 0.37).

GOAL
Return a smooth 2D trajectory from START to a TARGET consistent with the instruction, while avoiding obstacles inferred from the RGB image.

MAP INTERPRETATION (RGB-ONLY)
- There is no color-coded traversability. You MUST infer traversable vs obstacle regions from visual cues (e.g., occupied structures, walls, furniture-like shapes, cluttered regions, boundaries of open space).
- When uncertain, be conservative: route through visually open, continuous regions and keep margins from occupied structures.

INSTRUCTION GROUNDING
- Use the instruction to choose a target location on the map.
- If the instruction mentions an object category (chair/table/etc.), attempt to locate a plausible instance from the RGB map using shape/position/context cues.
- If multiple candidates exist, choose the one you judge most consistent with the instruction and overall scene layout.

TARGET REQUIREMENTS (NO FALLBACK; MUST ANSWER)
- You MUST always output a target with a valid (x,y) in [0.00, 1.00] x [0.00, 1.00].
- If you are uncertain about the exact target, output your BEST GUESS anyway.
- Indicate uncertainty in "notes" using short ASCII words (e.g., "best guess").

TRAJECTORY REQUIREMENTS (STRICT)
- Output exactly {self.num_waypoints} waypoints.
- Waypoint #1 must be at START (exact match preferred; otherwise within 0.02 L2 distance).
- Waypoint #{self.num_waypoints} must be at TARGET (exact match preferred; otherwise within 0.02 L2 distance).
- All waypoints must satisfy: 0.00 <= x <= 1.00 and 0.00 <= y <= 1.00.
- Avoid obstacles inferred from the RGB image (do not place waypoints on occupied structures).
- Prefer smooth paths with gentle curvature; avoid zig-zags.

WAYPOINT SPACING (SOFT)
- Prefer roughly even spacing when feasible. Collision avoidance has higher priority.

OUTPUT FORMAT (STRICT JSON ONLY)
Return ONLY a valid JSON object with double quotes and no trailing commas. No extra text.

Schema:
{{
  "target": {{"name": "<string>", "x": <float>, "y": <float>}},
  "trajectory": [
    {{"x": <float>, "y": <float>}},
    ...
  ],
  "notes": "<max 25 words; ASCII only; no line breaks>"
}}

NOTES (STRICT)
- Use only letters/numbers/spaces in notes (no punctuation).
- If target is uncertain, include "best guess".
- If obstacle inference is uncertain, include "conservative".
"""

    def draw_markers(self, image: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Draw start marker on image.

        NOTE: No goal marker - VLM must identify target from instruction.

        Args:
            image: Input image (H, W, 3) BGR format
            start: Start position in normalized coords [0,1]
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # Draw start marker (green circle) - VLM needs to know where robot is
        start_px = (int(start[0] * w), int(start[1] * h))
        cv2.circle(img, start_px, 8, (0, 255, 0), -1)  # Filled green
        cv2.circle(img, start_px, 8, (0, 0, 0), 2)      # Black border
        cv2.putText(img, "S", (start_px[0]-5, start_px[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # NO goal marker - VLM must identify target from instruction
        
        return img
    
    def prepare_image(self, image: np.ndarray, start: np.ndarray) -> str:
        """
        Prepare image for VLM with start marker only
        
        Args:
            image: Input image (H, W, 3) RGB format
            start: Start position [x, y] normalized [0,1]
            
        Returns:
            base64 encoded JPEG string
        """
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize to reasonable size
        target_size = 384
        h, w = img_bgr.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Add start marker only 
        img_marked = self.draw_markers(img_resized, start)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpeg', img_marked, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode()
    
    def generate_trajectory(self, image: np.ndarray, instruction: str,
                           start: np.ndarray) -> Dict[str, Any]:
        """
        Generate a trajectory from start to target (VLM identifies target from instruction)
        
        Args:
            image: Input image (H, W, 3) RGB format
            instruction: Navigation instruction (VLM uses this to find target)
            start: Start position [x, y] normalized [0,1]
            
        Returns:
            Dictionary with:
                - target: Identified target {name, x, y}
                - trajectory: List of [x, y] waypoints
                - reasoning: VLM's explanation
                - error: Error message if failed
        """
        img_base64 = self.prepare_image(image, start)
        image_bytes = base64.b64decode(img_base64)

        prompt_text = f"""This is a top-down semantic map of an indoor environment.

The GREEN dot marks the robot's current START position at approximately ({start[0]:.2f}, {start[1]:.2f}).

Navigation instruction: "{instruction}"

Your task:
1. Understand the instruction to identify the TARGET object/location in the image
2. Generate a trajectory of exactly {self.num_waypoints} waypoints from START to TARGET

The path should:
1. Start near the green marker (robot's position)
2. End at the TARGET location (you must identify from instruction)
3. Avoid all obstacles
4. Be smooth and efficient

Output JSON with target info and trajectory array."""

        
        max_retries = 2
        last_result: Optional[Dict[str, Any]] = None
        for attempt in range(max_retries):
            try:
                response_text = self._call_llm(image_bytes=image_bytes, prompt_text=prompt_text)

                result = self._parse_response(response_text, start)
                last_result = result

                # If we detect truncation/malformed JSON, retry.
                if result.get("error") and attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                if (not result.get("trajectory")) and attempt < max_retries - 1:
                    # Empty trajectory is usually a parsing failure; retry once or twice.
                    time.sleep(1)
                    continue

                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    # Final failure
                    return {
                        "target": None,
                        "trajectory": [],
                        "reasoning": "",
                        "raw_response": None,
                        "error": str(e),
                    }

        # If we exhausted retries and allow fallback, try to infer target and return straight line.
        if self.allow_fallback and last_result and last_result.get("raw_response"):
            inferred = self._infer_target_from_text(last_result["raw_response"])
            if inferred:
                goal = np.array([inferred["x"], inferred["y"]])
                return {
                    "target": inferred,
                    "trajectory": self._make_straight_line(start, goal),
                    "reasoning": "fallback straight line due to truncated output",
                    "raw_response": last_result.get("raw_response"),
                    "error": last_result.get("error") or "truncated output",
                }

        return last_result or {
            "target": None,
            "trajectory": [],
            "reasoning": "",
            "raw_response": None,
            "error": "Unknown failure",
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
            text = self._extract_json_text(response_text)
            if self._looks_truncated(text):
                # Let caller retry rather than trying to parse partial JSON.
                raise json.JSONDecodeError("Likely truncated JSON", text or "", max(0, len(text) - 1))

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                repaired = self._repair_json_text(text)
                data = json.loads(repaired)
            
            # Extract target
            if "target" in data and data["target"]:
                target = data["target"]
                if target.get("x") is not None and target.get("y") is not None:
                    x = float(np.clip(target["x"], 0, 1))
                    y = float(np.clip(target["y"], 0, 1))
                    result["target"] = {
                        "name": target.get("name", "unknown"),
                        "x": round(x, 2),
                        "y": round(y, 2),
                    }

            # Extract trajectory
            if "trajectory" in data and data["trajectory"]:
                for pt in data["trajectory"]:
                    if pt.get("x") is not None and pt.get("y") is not None:
                        x = float(np.clip(pt["x"], 0, 1))
                        y = float(np.clip(pt["y"], 0, 1))
                        result["trajectory"].append([round(x, 2), round(y, 2)])

            # Extract notes/reasoning (be robust)
            result["reasoning"] = data.get("notes", data.get("reasoning", ""))
            
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

            # If target exists but trajectory missing, optionally generate a fallback.
            if self.allow_fallback and result["target"] and len(result["trajectory"]) == 0:
                goal = np.array([result["target"]["x"], result["target"]["y"]])
                result["trajectory"] = self._make_straight_line(start, goal)
                result["reasoning"] = (result.get("reasoning") or "") + " fallback"
                
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
    from pathlib import Path
    
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
    planner = VLMTrajectoryPlanner(num_waypoints=16)
    
    # Generate trajectory (only provide start + instruction, NO goal)
    print('\nGenerating trajectory (VLM must identify target from instruction)...')
    start = np.array(ep['start'])
    
    result = planner.generate_trajectory(
        image=image,
        instruction=ep['instruction'],
        start=start,
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
