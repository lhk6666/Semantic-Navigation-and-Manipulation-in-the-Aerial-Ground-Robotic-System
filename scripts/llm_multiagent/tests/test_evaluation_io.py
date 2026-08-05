import json
import sys
import types as pytypes
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE / "planners") not in sys.path:
    sys.path.insert(0, str(HERE / "planners"))

from evaluation_io import (  # noqa: E402
    build_trajectory_archive_payload,
    load_episode_indices_file,
    load_evaluation_checkpoint,
    remove_error_rows_for_retry,
    require_api_key_from_env,
    save_evaluation_checkpoint,
    select_episode_indices,
    validate_backend_options,
)
from make_stratified_episode_subset import make_manifest  # noqa: E402
from semantic_map_planner import SemanticMapPlanner  # noqa: E402
from vlm_trajectory_planner import VLMTrajectoryPlanner  # noqa: E402


def test_index_manifest_and_text_selection(tmp_path):
    manifest = tmp_path / "indices.json"
    manifest.write_text(
        json.dumps({"dataset_indices": [7, 2, 9], "fraction": 0.1, "seed": 4}),
        encoding="utf-8",
    )
    indices, metadata = load_episode_indices_file(manifest, dataset_size=10)
    assert indices == [7, 2, 9]
    assert metadata["episode_indices_file_count"] == 3
    assert len(metadata["episode_indices_file_sha256"]) == 64

    selected, config = select_episode_indices(
        dataset_size=10,
        num_episodes=2,
        stride=None,
        episode_indices_file=str(manifest),
    )
    assert selected == [7, 2]
    assert config["selection_mode"] == "episode_indices_file"

    text_file = tmp_path / "indices.txt"
    text_file.write_text("1\n# comment\n5, 8\n", encoding="utf-8")
    assert load_episode_indices_file(text_file, dataset_size=10)[0] == [1, 5, 8]
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_episode_indices(
            dataset_size=10,
            num_episodes=None,
            stride=2,
            episode_indices_file=str(manifest),
        )


def test_archive_preserves_oob_and_marks_failures():
    meta = [
        {"sample_id": "a", "scene_id": "s", "start": [0.1, 0.2]},
        {"sample_id": "b", "scene_id": "s", "start": [0.3, 0.4]},
    ]
    results = [
        {"dataset_index": 0, "sample_id": "a", "scene_id": "s", "error": None},
        {"dataset_index": 1, "sample_id": "b", "scene_id": "s", "error": "API failed"},
    ]
    trajectories = [
        np.asarray([[0.1, 0.2], [1.2, -0.1]], dtype=np.float32),
        None,
    ]
    payload = build_trajectory_archive_payload(
        results=results, trajectories=trajectories, episode_meta=meta
    )
    assert payload["trajectory_length"].tolist() == [2, 1]
    assert payload["success"].tolist() == [True, False]
    np.testing.assert_allclose(payload["trajectory_xy_norm"][0, 1], [1.2, -0.1])
    np.testing.assert_allclose(payload["trajectory_xy_norm"][1, 0], [0.3, 0.4])
    assert np.isnan(payload["trajectory_xy_norm"][1, 1:]).all()


def test_checkpoint_resume_is_identity_and_config_strict(tmp_path):
    path = tmp_path / "checkpoint.json"
    config = {"backend": "genai", "model": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY"}
    meta = [{"sample_id": "a", "scene_id": "s", "start": [0.1, 0.2]}]
    results = [{"dataset_index": 0, "sample_id": "a", "scene_id": "s", "error": None}]
    trajectory = np.asarray([[0.1, 0.2], [0.5, 0.6]])
    save_evaluation_checkpoint(
        path, config=config, results=results, trajectories=[trajectory]
    )
    loaded_results, loaded_trajectories = load_evaluation_checkpoint(
        path,
        expected_config=config,
        episode_meta=meta,
        selected_indices=[0],
    )
    assert loaded_results == results
    np.testing.assert_allclose(loaded_trajectories[0], trajectory)
    with pytest.raises(ValueError, match="config differs"):
        load_evaluation_checkpoint(
            path,
            expected_config={**config, "model": "different"},
            episode_meta=meta,
            selected_indices=[0],
        )


def test_retry_errors_removes_failures_and_keeps_successes_unique():
    success_trajectory = np.asarray([[0.1, 0.2], [0.2, 0.3]])
    failed_trajectory = np.asarray([[0.4, 0.5]])
    results = [
        {"dataset_index": 3, "error": None, "sample_id": "ok"},
        {"dataset_index": 8, "error": "429 quota", "sample_id": "failed"},
    ]
    kept_results, kept_trajectories, retry_indices = remove_error_rows_for_retry(
        results, [success_trajectory, failed_trajectory]
    )
    assert [row["dataset_index"] for row in kept_results] == [3]
    assert retry_indices == [8]
    np.testing.assert_array_equal(kept_trajectories[0], success_trajectory)
    with pytest.raises(ValueError, match="Duplicate dataset_index"):
        remove_error_rows_for_retry([results[0], results[0]], [None, None])


def test_backend_config_records_env_name_not_secret(monkeypatch):
    secret = "test-secret-that-must-not-appear"
    monkeypatch.setenv("UNIT_TEST_GEMINI_KEY", secret)
    config = validate_backend_options(
        backend="genai",
        model="gemini-2.5-flash",
        api_key_env="UNIT_TEST_GEMINI_KEY",
    )
    assert config == {
        "backend": "genai",
        "model": "gemini-2.5-flash",
        "api_key_env_name": "UNIT_TEST_GEMINI_KEY",
    }
    assert require_api_key_from_env("UNIT_TEST_GEMINI_KEY") == secret
    assert secret not in json.dumps(config)
    with pytest.raises(ValueError, match="environment variable name") as caught:
        validate_backend_options(
            backend="genai", model="gemini-2.5-flash", api_key_env=secret
        )
    assert secret not in str(caught.value)


def test_genai_planners_use_locked_config_and_direct_system_instruction(monkeypatch):
    captured = {"configs": [], "keys": [], "http_options": [], "requests": []}

    class FakePart:
        @classmethod
        def from_bytes(cls, **kwargs):
            return ("image", kwargs)

        @classmethod
        def from_text(cls, **kwargs):
            return ("text", kwargs)

    class FakeContent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["configs"].append(kwargs)

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["requests"].append(kwargs)
            return pytypes.SimpleNamespace(text="{}")

    class FakeClient:
        def __init__(self, *, api_key, http_options):
            captured["keys"].append(api_key)
            captured["http_options"].append(http_options)
            self.models = FakeModels()

    class FakeRetryOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = pytypes.ModuleType("google.genai.types")
    fake_types.Part = FakePart
    fake_types.Content = FakeContent
    fake_types.GenerateContentConfig = FakeConfig
    fake_types.HttpRetryOptions = FakeRetryOptions
    fake_types.HttpOptions = FakeHttpOptions
    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types
    try:
        import google as google_module
    except ImportError:
        google_module = pytypes.ModuleType("google")
        google_module.__path__ = []
        monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setattr(google_module, "genai", fake_genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setenv("UNIT_TEST_GEMINI_KEY", "private-value")

    direct = VLMTrajectoryPlanner(
        backend="genai",
        model="gemini-2.5-flash",
        api_key_env="UNIT_TEST_GEMINI_KEY",
        allow_fallback=False,
    )
    direct._call_llm(b"jpeg", "episode prompt")
    semantic = SemanticMapPlanner(
        backend="genai",
        model="gemini-2.5-flash",
        api_key_env="UNIT_TEST_GEMINI_KEY",
    )
    semantic._call_llm(b"jpeg", "episode prompt")

    assert captured["keys"] == ["private-value", "private-value"]
    assert all(option.kwargs["timeout"] == 120000 for option in captured["http_options"])
    assert all(
        option.kwargs["retry_options"].kwargs["attempts"] == 1
        for option in captured["http_options"]
    )
    assert all(config["temperature"] == 0.0 for config in captured["configs"])
    assert all(config["top_p"] == 1.0 for config in captured["configs"])
    assert all(config["max_output_tokens"] == 8192 for config in captured["configs"])
    assert captured["configs"][0]["system_instruction"] == direct._get_system_prompt()
    assert captured["configs"][1]["system_instruction"] == semantic._get_system_prompt()
    assert "private-value" not in repr(direct.__dict__)
    assert "private-value" not in repr(semantic.__dict__)

    direct.allow_fallback = True
    fallback = direct._parse_response(
        '{"target":{"name":"chair","x":0.5,"y":0.6},"trajectory":[]}',
        np.asarray([0.1, 0.2]),
    )
    assert fallback["used_fallback"] is True
    assert len(fallback["trajectory"]) == direct.num_waypoints

    direct_attempts = []
    semantic_attempts = []
    direct.before_request = lambda: direct_attempts.append("request")
    semantic.before_request = lambda: semantic_attempts.append("request")
    planner_module = sys.modules[VLMTrajectoryPlanner.__module__]
    monkeypatch.setattr(planner_module.time, "sleep", lambda _seconds: None)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    direct.generate_trajectory(image, "go to the chair", np.asarray([0.1, 0.2]))
    semantic.analyze_scene(image, "go to the chair", require_target=True)
    assert direct_attempts == ["request", "request"]
    assert semantic_attempts == ["request", "request"]
    retry_text = captured["requests"][-1]["contents"][0].kwargs["parts"][1][1]["text"]
    assert "target.center" in retry_text
    assert "target.bbox" in retry_text
    assert "target.x" not in retry_text


def test_stratified_manifest_uses_round_half_up_hamilton_and_is_reproducible():
    meta = []
    for exact_scene, count in (
        ("scene0700_00", 27),
        ("scene0700_01", 23),
        ("scene0700_02", 24),
        ("matterport", 15),
    ):
        meta.extend({"scene_id": exact_scene} for _ in range(count))
    first = make_manifest(meta, fraction=0.1, seed=17, episode_meta_sha256="abc")
    second = make_manifest(meta, fraction=0.1, seed=17, episode_meta_sha256="abc")
    assert first == second
    assert first["source_scenes"]["scene0700"]["selected_count"] == 7
    exact = first["source_scenes"]["scene0700"]["exact_scenes"]
    assert sum(row["selected_count"] for row in exact.values()) == 7
    assert first["source_scenes"]["matterport"]["selected_count"] == 2
    assert first["dataset_indices"] == sorted(first["dataset_indices"])
