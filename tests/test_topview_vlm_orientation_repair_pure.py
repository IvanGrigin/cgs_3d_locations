from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from src import topview_vlm_orientation_repair as repair


def api_response(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def sample_scene() -> dict:
    return {
        "room": {"type": "office"},
        "objects": [
            {
                "id": "chair_1",
                "category": "chair",
                "name": "Office chair",
                "yaw_deg": 5.0,
                "position": [0.0, 0.0, 0.0],
                "size": [0.5, 0.5, 0.8],
                "meta": {"support_group": "desk_a"},
            },
            {
                "id": "desk_1",
                "category": "desk",
                "name": "Writing desk",
                "rotation": {"z": 1.5707963267948966},
                "position": [0.0, 1.0, 0.0],
                "dimensions": [1.2, 0.7, 0.75],
                "meta": {"support_group": "desk_a"},
            },
            {
                "id": "armchair_1",
                "category": "armchair",
                "name": "Reading armchair",
                "rotation_euler": [0.0, 0.0, 3.141592653589793],
                "position_m": {"x": 2.0, "y": 2.0},
                "width": 0.8,
                "depth": 0.8,
            },
        ],
    }


def test_collect_scene_objects_filters_and_sets_yaw():
    scene = sample_scene()
    refs = repair.collect_scene_objects(scene)
    by_id = {ref.object_id: ref for ref in refs}

    assert list(by_id) == ["chair_1", "desk_1", "armchair_1"]
    assert by_id["chair_1"].yaw_deg == 5.0
    assert by_id["desk_1"].yaw_deg == 90.0
    assert by_id["armchair_1"].yaw_deg == 180.0
    assert by_id["desk_1"].position_xy == (0.0, 1.0)
    assert by_id["desk_1"].size_xy == (1.2, 0.7)
    assert repair.is_chair_like(by_id["chair_1"]) is True
    assert repair.is_chair_like(by_id["armchair_1"]) is False
    assert repair.is_chair_like(by_id["armchair_1"], include_armchairs=True) is True
    assert repair.is_table_like(by_id["desk_1"]) is True
    assert repair.filter_target_objects(refs, scope="chairs") == [by_id["chair_1"]]
    assert repair.filter_target_objects(refs, scope="chairs", include_armchairs=True) == [by_id["chair_1"], by_id["armchair_1"]]
    with pytest.raises(ValueError):
        repair.filter_target_objects(refs, scope="bad")

    updated, report = repair.set_scene_object_yaws(scene, {"chair_1": 95, "missing": 180})
    assert updated["objects"][0]["yaw_deg"] == 95.0
    assert report["applied"][0]["field"] == "yaw_deg"
    assert report["skipped"] == [{"object_id": "missing", "reason": "object_id_not_found"}]


def test_prompts_and_response_parsers_validate_labels():
    refs = repair.collect_scene_objects(sample_scene())
    targets = [refs[0]]
    label_map = {"C1": "chair_1"}
    prompt = repair.build_vlm_prompt(sample_scene(), targets, refs, scope="chairs", label_by_object_id={"chair_1": "C1"})
    assert "chair-only" in prompt
    assert '"C1"' in prompt
    assert "Office chair" in prompt

    all_prompt = repair.build_vlm_prompt(sample_scene(), refs, refs, scope="all")
    assert "orientation-only review" in all_prompt
    assert "set_yaw" in all_prompt

    fenced = repair._extract_json_text('```json\n{"summary": "ok"}\n```')
    assert fenced == '{"summary": "ok"}'

    parsed = repair.parse_vlm_response(
        api_response(
            {
                "summary": "one fix",
                "objects": [
                    {"label_id": "C1", "action": "set_yaw", "target_yaw_deg": 455, "confidence": 1.2, "reason": "faces desk"}
                ],
            }
        ),
        label_map=label_map,
    )
    assert parsed.summary == "one fix"
    assert parsed.decisions[0].object_id == "chair_1"
    assert parsed.decisions[0].target_yaw_deg == 95.0
    assert parsed.decisions[0].confidence == 1.0

    err_label_map = {f"C{i}": "chair_1" for i in range(1, 7)}
    review, errors = repair.parse_vlm_judge_response(
        api_response(
            {
                "summary": "judge",
                "objects": [
                    {"label_id": "C1", "status": "wrong", "confidence": 0.9, "relation": "face_desk", "reason": "back to desk"}
                ],
            }
        ),
        label_map=label_map,
        target_by_id={"chair_1": refs[0]},
    )
    assert not errors
    assert review.decisions[0].status == "wrong"
    assert review.decisions[0].object_name == "Office chair"

    _review, errors = repair.parse_vlm_judge_response(
        api_response(
            {
                "summary": "bad",
                "objects": [
                    {"label_id": "C1", "action": "rotate_90", "confidence": 0.9},
                    {"label_id": "C2", "status": "ok", "confidence": 0.9},
                ],
            }
        ),
        label_map=label_map,
        target_by_id={"chair_1": refs[0]},
    )
    assert {err["reason"] for err in errors} >= {"forbidden_relative_action", "label_id_not_allowed", "missing_label_decisions"}


def test_response_parser_error_matrix_and_decision_skip_branches():
    refs = repair.collect_scene_objects(sample_scene())
    label_map = {"C1": "chair_1"}
    with pytest.raises(ValueError, match="no choices"):
        repair.parse_vlm_judge_response({}, label_map=label_map, target_by_id={"chair_1": refs[0]})
    with pytest.raises(ValueError, match="does not contain a JSON object"):
        repair.parse_vlm_judge_response({"choices": [{"message": {"content": "plain text"}}]}, label_map=label_map, target_by_id={"chair_1": refs[0]})

    err_label_map = {f"C{i}": "chair_1" for i in range(1, 7)}
    review, errors = repair.parse_vlm_judge_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": json.dumps(
                                {
                                    "summary": "bad entries",
                                    "objects": [
                                        "not-dict",
                                        {"label_id": "", "object_id": "chair_1"},
                                        {"label_id": "C9", "status": "ok"},
                                        {"label_id": "C1", "object_id": "other", "status": "ok"},
                                        {"label_id": "C2", "action": "keep", "target_yaw_deg": 90},
                                        {"label_id": "C3", "action": "set_yaw"},
                                        {"label_id": "C4", "status": "bad"},
                                        {"label_id": "C5", "status": "wrong", "confidence": 0.8},
                                        {"label_id": "C5", "status": "wrong", "confidence": 0.9},
                                    ],
                                }
                            )},
                        ],
                    }
                }
            ]
        },
        label_map=err_label_map,
        target_by_id={"chair_1": refs[0]},
    )
    reasons = {err["reason"] for err in errors}
    assert {
        "object_entry_not_dict",
        "missing_label_id",
        "label_id_not_allowed",
        "object_id_mismatch_for_label",
        "keep_with_target_yaw",
        "set_yaw_missing_target_yaw",
        "invalid_status",
        "missing_label_decisions",
    } <= reasons
    assert review.decisions[0].object_id == "chair_1"

    assert repair._decision_target_yaw(repair.OrientationDecision("x", "x", "rotate_90", None, None, 1.0, ""), 10) == 100.0
    assert repair._decision_target_yaw(repair.OrientationDecision("x", "x", "rotate_minus_90", None, None, 1.0, ""), 10) == 280.0
    assert repair._decision_target_yaw(repair.OrientationDecision("x", "x", "rotate_180", None, None, 1.0, ""), 10) == 190.0
    assert repair._decision_target_yaw(repair.OrientationDecision("x", "x", "wrong", None, 45, 1.0, ""), None) == 45.0

    decisions = [
        repair.OrientationDecision("chair_1", "chair", "wrong", None, None, 0.95, "missing target"),
        repair.OrientationDecision("chair_1", "chair", "set_yaw", None, 270, 0.95, "too far"),
        repair.OrientationDecision("chair_1", "chair", "set_yaw", None, 90, 0.2, "low confidence"),
    ]
    _updated, report = repair.apply_orientation_decisions(
        sample_scene(),
        decisions,
        min_confidence=0.7,
        max_delta_deg=30,
        snap_step_deg=90,
        keep_large_delta_only_if_confident=False,
    )
    assert [row["reason"] for row in report["skipped"]] == ["missing_target_yaw_deg", "delta_too_large", "low_confidence"]


def test_variant_parser_and_provider_none_runner(tmp_path):
    label_map = {"C1": "chair_1", "C2": "chair_2"}
    selection, errors = repair.parse_variant_selection_response(
        api_response(
            {
                "summary": "selected",
                "objects": [
                    {"label_id": "chair 1", "best_variant_id": "90 degrees", "status": "selected", "confidence": 0.8},
                    {"label_id": "C2", "best_variant_id": None, "status": "unclear", "confidence": 0.2},
                ],
            }
        ),
        label_map=label_map,
        variant_ids={"offset_000", "offset_090"},
    )
    assert not errors
    assert selection["objects"][0]["label_id"] == "C1"
    assert selection["objects"][0]["best_variant_id"] == "offset_090"
    assert selection["objects"][1]["status"] == "unclear"

    _selection, errors = repair.parse_variant_selection_response(
        api_response({"summary": "bad", "objects": [{"label_id": "C1", "best_variant_id": "offset_999"}]}),
        label_map=label_map,
        variant_ids={"offset_000"},
    )
    assert {err["reason"] for err in errors} >= {"variant_id_not_allowed", "missing_label_selections"}

    sheet = tmp_path / "contact.png"
    sheet.write_bytes(b"not really png")
    report = repair.run_topview_vlm_variant_selection(
        contact_sheet_path=sheet,
        label_map={"C1": "chair_1"},
        variants=[{"variant_id": "offset_000", "offset_deg": 0, "target_yaws_deg": {"chair_1": 0}}],
        out_prompt_path=tmp_path / "prompt.txt",
        out_review_path=tmp_path / "review.json",
        out_report_path=tmp_path / "report.json",
        provider="none",
        model=None,
        min_confidence=0.5,
    )
    assert report["stop_reason"] == "unclear_vlm_response"
    assert (tmp_path / "prompt.txt").is_file()
    assert repair.image_to_data_url(sheet).startswith("data:image/png;base64,")
    assert repair.image_to_base64(sheet)


def test_apply_orientation_and_geometry_decisions():
    scene = sample_scene()
    decisions = [
        repair.OrientationDecision("chair_1", "Office chair", "set_yaw", None, 92, 0.91, "face desk"),
        repair.OrientationDecision("desk_1", "Writing desk", "set_yaw", None, 180, 0.95, "not a chair"),
        repair.OrientationDecision("chair_1", "Office chair", "keep", None, None, 1.0, "already ok"),
        repair.OrientationDecision("missing", "Missing", "set_yaw", None, 0, 1.0, "missing"),
    ]
    updated, report = repair.apply_orientation_decisions(scene, decisions, min_confidence=0.7, snap_step_deg=90)
    assert updated["objects"][0]["yaw_deg"] == 90.0
    assert report["counts"]["applied"] == 1
    assert {row["reason"] for row in report["skipped"]} >= {"object_id_not_allowed_or_not_found", "action_keep"}

    assert repair.quantize_yaw(450, 90) == 1
    refs = repair.collect_scene_objects(scene)
    state = repair._room_state_key(refs, 90)
    assert repair._state_in_history(state, [state]) is True

    chair = next(ref for ref in refs if ref.object_id == "chair_1")
    target_yaw, solver = repair._geometry_yaw_for_chair(chair, refs, visual_front_offset_deg=0.0, snap_step_deg=90)
    assert target_yaw == 90.0
    assert solver["table_object_id"] == "desk_1"

    judge_decisions = [
        repair.OrientationDecision(
            object_id="chair_1",
            object_name="Office chair",
            action="wrong",
            clockwise_delta_deg=None,
            target_yaw_deg=None,
            confidence=0.9,
            reason="backrest toward table",
            label_id="C1",
            status="wrong",
            relation="face_desk",
        )
    ]
    repaired, repair_report = repair.apply_chair_judge_geometry_decisions(
        scene,
        judge_decisions,
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        snap_step_deg=90,
        label_map={"C1": "chair_1"},
    )
    assert repaired["objects"][0]["yaw_deg"] == 90.0
    assert repair_report["stop_reason"] == "geometry_applied"
    assert repair_report["counts"]["applied"] == 1


def test_provider_http_helpers_are_mocked(tmp_path, monkeypatch):
    image = tmp_path / "view.jpg"
    image.write_bytes(b"jpg")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("TOPVIEW_VLM_OLLAMA_URL", "http://ollama.test")

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    requests = []

    def fake_urlopen(request, timeout=0):
        requests.append((request.full_url, timeout, json.loads(request.data.decode("utf-8"))))
        if request.full_url.endswith("/api/chat"):
            return FakeResponse({"message": {"content": '{"summary": "ollama", "objects": []}'}})
        return FakeResponse({"choices": [{"message": {"content": '{"summary": "http", "objects": []}'}}]})

    monkeypatch.setattr(repair.urllib.request, "urlopen", fake_urlopen)

    openai = repair.call_openai_compatible_vlm(provider="openai", model=None, prompt="p", image_path=image)
    openrouter = repair.call_openai_compatible_vlm(provider="openrouter", model="router-model", prompt="p", image_path=image)
    ollama = repair.call_ollama_vlm(model=None, prompt="p", image_path=image)
    ollama_multi = repair.call_ollama_vlm_multi(model="m", prompt="p", image_paths=[image])

    assert openai["choices"][0]["message"]["content"].startswith("{")
    assert openrouter["choices"][0]["message"]["content"].startswith("{")
    assert ollama["choices"][0]["message"]["content"].startswith("{")
    assert ollama_multi["choices"][0]["message"]["content"].startswith("{")
    assert any(url == "https://api.openai.com/v1/chat/completions" for url, _timeout, _payload in requests)
    assert any(url == "https://openrouter.ai/api/v1/chat/completions" for url, _timeout, _payload in requests)
    assert any(url == "http://ollama.test/api/chat" for url, _timeout, _payload in requests)

    with pytest.raises(ValueError):
        repair.call_ollama_vlm_multi(model=None, prompt="p", image_paths=[])

    def failing_urlopen(_request, timeout=0):
        del timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(repair.urllib.request, "urlopen", failing_urlopen)
    with pytest.raises(RuntimeError, match="VLM request failed"):
        repair.call_openai_compatible_vlm(provider="openai", model=None, prompt="p", image_path=image)
    with pytest.raises(RuntimeError, match="Ollama VLM request failed"):
        repair.call_ollama_vlm(model=None, prompt="p", image_path=image)


def test_low_level_extractors_dispatch_and_cycle_guards(tmp_path, monkeypatch, capsys):
    scene_path = tmp_path / "scene.json"
    repair.write_json(scene_path, sample_scene())
    assert repair.load_json(scene_path)["room"]["type"] == "office"

    assert repair._as_float(True) is None
    assert repair._as_float("nan") is None
    assert repair._angle_delta_deg(350, 10) == 20
    assert repair._extract_yaw_deg({"transform": {"rotation_euler": {"z_deg": -90}}}) == 270.0
    assert repair._extract_position_xy({"transform": {"location": {"x": 1, "y": 2}}}) == (1.0, 2.0)
    assert repair._extract_size_xy({"aabb": {"min": {"x": 1, "y": 2}, "max": {"x": 4, "y": 6}}}) == (3.0, 4.0)
    assert repair._extract_object_id({}, "fallback") == "fallback"
    assert repair._extract_category({}) == "unknown"
    assert repair._extract_name({}) == ""

    obj = {"transform": {"rotation": {"z_deg": 0}}}
    assert repair._set_yaw_deg(obj, 450) == "transform.rotation.z_deg"
    assert obj["transform"]["rotation"]["z_deg"] == 90.0
    obj = {"rotation_euler": {"yaw": 0.0}}
    assert repair._set_yaw_deg(obj, 180) == "rotation_euler.yaw"
    assert obj["rotation_euler"]["yaw"] == pytest.approx(3.141592653589793)
    obj = {}
    assert repair._set_yaw_deg(obj, 45) == "yaw_deg/rotation_deg/yaw_rad"
    assert obj["yaw_deg"] == 45.0

    assert list(repair._iter_candidate_lists({"nested": {"objects": [1, {"id": "x"}]}}))
    assert repair.collect_scene_objects({"other": [{"id": "a"}]}, max_objects=1)[0].object_id == "a"

    review_path = tmp_path / "review.json"
    repair.write_json(
        review_path,
        {
            "summary": "manual",
            "objects": [
                {"object_id": "chair_1", "action": "rotate_clockwise", "rotate_clockwise_deg": 90, "confidence": 0.9},
                {"object_id": "", "action": "set_yaw", "target_yaw_deg": 0},
            ],
        },
    )
    review = repair.load_review_from_file(review_path)
    assert review.decisions[0].clockwise_delta_deg == 90.0

    choices_path = tmp_path / "choices.json"
    repair.write_json(choices_path, api_response({"summary": "api", "objects": [{"object_id": "chair_1", "action": "keep"}]}))
    assert repair.load_review_from_file(choices_path).summary == "api"

    decision = repair.OrientationDecision("chair_1", "chair", "rotate_clockwise", 90, None, 0.95, "turn")
    updated, report = repair.apply_orientation_decisions(sample_scene(), [decision], min_confidence=0.7, snap_step_deg=90)
    assert updated["objects"][0]["yaw_deg"] == 270.0
    assert report["counts"]["applied"] == 1

    low = repair.OrientationDecision("chair_1", "chair", "set_yaw", None, 180, 0.8, "large")
    _updated, report = repair.apply_orientation_decisions(sample_scene(), [low], min_confidence=0.7, snap_step_deg=90)
    assert report["skipped"][0]["reason"] == "large_delta_requires_higher_confidence"

    refs = repair.collect_scene_objects(sample_scene())
    chair = refs[0]
    assert repair._nearest_table_for_chair(chair, refs)[0].object_id == "desk_1"
    no_pos = repair.SceneObjectRef("c", "chair", "chair", (), {}, 0, (None, None), (None, None))
    assert repair._nearest_table_for_chair(no_pos, refs) == (None, None)
    assert repair._angle_deg_from_a_to_b(0, 0, 0, 1) == 90.0
    history = {}
    repair._append_yaw_history(history, "chair_1", 90)
    repair._append_yaw_history(history, "chair_1", 90.0001)
    assert history["chair_1"] == [90.0]

    wrong = repair.OrientationDecision("chair_1", "chair", "wrong", None, None, 0.95, "wrong", label_id="C1", status="wrong")
    _updated, report = repair.apply_chair_judge_geometry_decisions(
        sample_scene(),
        [wrong],
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        snap_step_deg=90,
        label_map={"C1": "chair_1"},
        yaw_history={"chair_1": [90]},
    )
    assert report["stop_reason"] == "yaw_cycle_detected"

    _updated, report = repair.apply_chair_judge_geometry_decisions(
        sample_scene(),
        [wrong],
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        snap_step_deg=90,
        label_map={"C1": "chair_1"},
        repair_counts={"chair_1": 1},
    )
    assert report["stop_reason"] == "object_repair_limit_reached"

    monkeypatch.setattr(repair, "call_ollama_vlm", lambda **_kwargs: {"ok": "ollama"})
    monkeypatch.setattr(repair, "call_openai_compatible_vlm", lambda **_kwargs: {"ok": "http"})
    assert repair.call_vlm_json(provider="ollama", model=None, prompt="p", image_path=tmp_path / "x.png") == {"ok": "ollama"}
    assert repair.call_vlm_json(provider="openai", model=None, prompt="p", image_path=tmp_path / "x.png") == {"ok": "http"}
    with pytest.raises(ValueError):
        repair.call_vlm_json(provider="none", model=None, prompt="p", image_path=tmp_path / "x.png")
    monkeypatch.setattr(repair, "call_ollama_vlm_multi", lambda **_kwargs: {"ok": "multi"})
    assert repair.call_vlm_json_multi(provider="ollama", model=None, prompt="p", image_paths=[tmp_path / "x.png"]) == {"ok": "multi"}
    assert repair.call_vlm_json_multi(provider="openai", model=None, prompt="p", image_paths=[tmp_path / "x.png"]) == {"ok": "http"}
    with pytest.raises(ValueError):
        repair.call_vlm_json_multi(provider="openai", model=None, prompt="p", image_paths=[tmp_path / "a.png", tmp_path / "b.png"])

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        repair._provider_config("openai", None)
    for key in list(repair.os.environ):
        if key == "OPENROUTER_API_KEY" or key == "ivangrigin_OPENROUTER_API_KEY" or key.startswith("ivangrigin_OPENROUTER_API_KEY_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ivangrigin_OPENROUTER_API_KEY_2", "k2")
    monkeypatch.setenv("ivangrigin_OPENROUTER_API_KEY_1", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k0")
    assert repair._openrouter_keys_from_env() == ["k0", "k1", "k2"]
    assert repair._provider_config("openrouter", "m")[2] == "m"
    with pytest.raises(ValueError):
        repair._provider_config("bad", None)

    args = repair.build_arg_parser().parse_args(
        [
            "--scene",
            str(scene_path),
            "--topview-image",
            str(tmp_path / "view.png"),
            "--out-scene",
            str(tmp_path / "out.json"),
            "--out-review",
            str(tmp_path / "review_out.json"),
            "--out-report",
            str(tmp_path / "report.json"),
            "--provider",
            "none",
            "--target-scope",
            "all",
            "--no-apply",
        ]
    )
    assert args.provider == "none"
    (tmp_path / "view.png").write_bytes(b"png")
    assert repair.main(
        [
            "--scene",
            str(scene_path),
            "--topview-image",
            str(tmp_path / "view.png"),
            "--out-scene",
            str(tmp_path / "main_out.json"),
            "--out-review",
            str(tmp_path / "main_review.json"),
            "--out-report",
            str(tmp_path / "main_report.json"),
            "--provider",
            "none",
            "--target-scope",
            "all",
            "--no-apply",
        ]
    ) == 0
    assert "stop_reason" in capsys.readouterr().out


def test_run_topview_orientation_repair_main_branches(tmp_path, monkeypatch):
    image = tmp_path / "view.png"
    image.write_bytes(b"png")

    no_target_scene = tmp_path / "no_target.json"
    no_target_scene.write_text(json.dumps({"objects": [{"id": "lamp", "category": "lamp"}]}), encoding="utf-8")
    report = repair.run_topview_vlm_orientation_repair(
        scene_path=no_target_scene,
        image_path=image,
        out_scene_path=tmp_path / "no_target.out.json",
        out_review_path=tmp_path / "no_target.review.json",
        out_report_path=tmp_path / "no_target.report.json",
        provider="none",
        model=None,
        max_objects=100,
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        max_delta_deg=180,
        snap_step_deg=90,
    )
    assert report["stop_reason"] == "no_target_objects"

    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(sample_scene()), encoding="utf-8")
    label_map = tmp_path / "labels.json"
    label_map.write_text(json.dumps({"C1": "chair_1"}), encoding="utf-8")
    invalid_review = tmp_path / "invalid_review.json"
    invalid_review.write_text(json.dumps({"summary": "bad", "objects": [{"label_id": "C2", "status": "ok", "confidence": 1.0}]}), encoding="utf-8")
    report = repair.run_topview_vlm_orientation_repair(
        scene_path=scene_path,
        image_path=image,
        out_scene_path=tmp_path / "invalid.out.json",
        out_review_path=tmp_path / "invalid.review.json",
        out_report_path=tmp_path / "invalid.report.json",
        provider="none",
        model=None,
        max_objects=100,
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        max_delta_deg=180,
        snap_step_deg=90,
        target_label_map_path=label_map,
        review_json_path=invalid_review,
    )
    assert report["stop_reason"] == "invalid_vlm_response"
    assert report["counts"]["validation_errors"] >= 1

    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"summary": "apply off", "objects": [{"object_id": "chair_1", "action": "set_yaw", "target_yaw_deg": 180, "confidence": 0.95}]}),
        encoding="utf-8",
    )
    report = repair.run_topview_vlm_orientation_repair(
        scene_path=scene_path,
        image_path=image,
        out_scene_path=tmp_path / "apply_off.out.json",
        out_review_path=tmp_path / "apply_off.review.json",
        out_report_path=tmp_path / "apply_off.report.json",
        provider="none",
        model=None,
        max_objects=100,
        target_scope="all",
        include_armchairs=True,
        min_confidence=0.7,
        max_delta_deg=180,
        snap_step_deg=90,
        review_json_path=review,
        apply=False,
    )
    assert report["apply"] is False

    chair_review = tmp_path / "chair_review.json"
    chair_review.write_text(
        json.dumps({"summary": "judge", "objects": [{"label_id": "C1", "status": "wrong", "confidence": 0.95, "relation": "face_desk"}]}),
        encoding="utf-8",
    )
    report = repair.run_topview_vlm_orientation_repair(
        scene_path=scene_path,
        image_path=image,
        out_scene_path=tmp_path / "chair_apply_off.out.json",
        out_review_path=tmp_path / "chair_apply_off.review.json",
        out_report_path=tmp_path / "chair_apply_off.report.json",
        provider="none",
        model=None,
        max_objects=100,
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        max_delta_deg=180,
        snap_step_deg=90,
        target_label_map_path=label_map,
        review_json_path=chair_review,
        apply=False,
    )
    assert report["apply"] is False

    monkeypatch.setattr(
        repair,
        "call_ollama_vlm",
        lambda **_kwargs: api_response(
            {"summary": "judge", "objects": [{"label_id": "C1", "status": "wrong", "confidence": 0.95, "relation": "face_desk"}]}
        ),
    )
    report = repair.run_topview_vlm_orientation_repair(
        scene_path=scene_path,
        image_path=image,
        out_scene_path=tmp_path / "ollama.out.json",
        out_review_path=tmp_path / "ollama.review.json",
        out_report_path=tmp_path / "ollama.report.json",
        provider="ollama",
        model="vision",
        max_objects=100,
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        max_delta_deg=180,
        snap_step_deg=90,
        target_label_map_path=label_map,
    )
    assert report["provider"] == "ollama"
    assert report["counts"]["applied"] == 1
    assert json.loads((tmp_path / "ollama.out.json").read_text(encoding="utf-8"))["objects"][0]["yaw_deg"] == 90.0


def test_topview_parser_geometry_and_provider_edge_branches(tmp_path, monkeypatch):
    assert repair._as_float("bad") is None
    assert repair.norm_angle_deg(-10) == 350.0
    assert repair._extract_json_text('prefix {"ok": true} suffix') == '{"ok": true}'
    assert repair._extract_yaw_deg({"yaw_rad": 3.141592653589793}) == 180.0
    assert repair._extract_yaw_deg({"rotation": {"z_deg": 270}}) == 270.0
    assert repair._extract_yaw_deg({"rotation_euler": {"yaw": 3.141592653589793 / 2}}) == 90.0

    obj = {"yaw_rad": 0.0}
    assert repair._set_yaw_deg(obj, 90) == "yaw_rad"
    assert obj["yaw_rad"] == pytest.approx(3.141592653589793 / 2)
    obj = {"rotation": {"z": 0.0}}
    assert repair._set_yaw_deg(obj, 180) == "rotation.z"
    obj = {"rotation_euler": {"z_deg": 0.0}}
    assert repair._set_yaw_deg(obj, 45) == "rotation_euler.z_deg"
    obj = {"rotation_euler": [0.0, 0.0, 0.0]}
    assert repair._set_yaw_deg(obj, 90) == "rotation_euler[2]"

    with pytest.raises(ValueError, match="no choices"):
        repair.parse_vlm_response({}, label_map={})
    parsed = repair.parse_vlm_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "summary": "list content",
                                        "objects": [
                                            "bad",
                                            {"label_id": "C1", "action": "keep", "confidence": "bad"},
                                            {"object_id": "chair_1", "action": "rotate_clockwise", "rotate_clockwise_deg": -90, "confidence": 0.5},
                                        ],
                                    }
                                ),
                            }
                        ]
                    }
                }
            ]
        },
        label_map={"C1": "chair_1"},
    )
    assert len(parsed.decisions) == 2
    assert parsed.decisions[1].clockwise_delta_deg == 270.0
    empty = repair.parse_vlm_response(api_response({"summary": "empty", "objects": {"bad": True}}), label_map={})
    assert empty.decisions == []

    refs = repair.collect_scene_objects(sample_scene())
    review, errors = repair.parse_vlm_judge_response(
        api_response({"summary": "bad", "objects": {"not": "a list"}}),
        label_map={"C1": "chair_1"},
        target_by_id={"chair_1": refs[0]},
    )
    assert review.decisions == []
    assert errors[0]["reason"] == "objects_not_list"
    _review, errors = repair.parse_vlm_judge_response(
        api_response({"summary": "bad action", "objects": [{"label_id": "C1", "action": "spin", "status": "ok"}]}),
        label_map={"C1": "chair_1"},
        target_by_id={"chair_1": refs[0]},
    )
    assert errors[0]["reason"] == "invalid_action"

    with pytest.raises(ValueError, match="no choices"):
        repair.parse_variant_selection_response({}, label_map={"C1": "chair_1"}, variant_ids={"offset_000"})
    selection, errors = repair.parse_variant_selection_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "summary": "variants",
                                        "objects": [
                                            "bad",
                                            {"label_id": "chair 2", "best_variant_id": "variant offset_180", "confidence": 2.0},
                                            {"label_id": "C2", "best_variant_id": "offset_000"},
                                            {"label_id": "C9", "best_variant_id": "offset_000"},
                                        ],
                                    }
                                ),
                            }
                        ]
                    }
                }
            ]
        },
        label_map={"C1": "chair_1", "C2": "chair_2"},
        variant_ids={"offset_000", "offset_180"},
    )
    reasons = {err["reason"] for err in errors}
    assert {"object_entry_not_dict", "duplicate_label_id", "label_id_not_allowed", "missing_label_selections"} <= reasons
    assert selection["objects"][0]["label_id"] == "C2"
    selection, errors = repair.parse_variant_selection_response(
        api_response({"summary": "bad", "objects": {"bad": True}}),
        label_map={"C1": "chair_1"},
        variant_ids={"offset_000"},
    )
    assert selection["objects"] == []
    assert {err["reason"] for err in errors} == {"objects_not_list", "missing_label_selections"}

    status_decisions = [
        repair.OrientationDecision("chair_1", "chair", "keep", None, None, 0.9, "ok", label_id="C1", status="ok"),
        repair.OrientationDecision("chair_1", "chair", "wrong", None, None, 0.1, "low", label_id="C1", status="wrong"),
        repair.OrientationDecision("chair_1", "chair", "unclear", None, None, 0.9, "unclear", label_id="C1", status="unclear"),
        repair.OrientationDecision("chair_1", "chair", "mystery", None, None, 0.9, "bad", label_id="C1", status="mystery"),
    ]
    _scene, report = repair.apply_chair_judge_geometry_decisions(
        sample_scene(),
        status_decisions,
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        snap_step_deg=90,
        label_map={"C1": "chair_1"},
    )
    assert report["stop_reason"] == "unclear_vlm_response"
    assert {row["reason"] for row in report["unclear"]} >= {"low_confidence", "status_unclear", "unsupported_status"}
    no_table_scene = {"objects": [{"id": "chair_1", "category": "chair", "yaw_deg": 0, "position": [0, 0], "size": [0.5, 0.5]}]}
    _scene, report = repair.apply_chair_judge_geometry_decisions(
        no_table_scene,
        [repair.OrientationDecision("chair_1", "chair", "wrong", None, None, 0.9, "bad", label_id="C1", status="wrong")],
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        snap_step_deg=90,
        label_map={"C1": "chair_1"},
    )
    assert report["stop_reason"] == "invalid_vlm_response"
    assert report["skipped"][0]["reason"] == "geometry_solver_failed"
    _scene, report = repair.apply_chair_judge_geometry_decisions(
        sample_scene(),
        [repair.OrientationDecision("chair_1", "chair", "wrong", None, None, 0.9, "bad", label_id="C1", status="wrong")],
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        snap_step_deg=90,
        label_map={"C1": "chair_1"},
        room_state_history=[[["chair_1", 1]]],
    )
    assert report["stop_reason"] == "yaw_cycle_detected"
    assert report["skipped"][0]["reason"] == "room_state_cycle_detected"
    original_get_by_path = repair._get_by_path
    monkeypatch.setattr(repair, "_get_by_path", lambda _root, _path: None)
    _scene, report = repair.apply_chair_judge_geometry_decisions(
        sample_scene(),
        [repair.OrientationDecision("chair_1", "chair", "wrong", None, None, 0.9, "bad", label_id="C1", status="wrong")],
        target_scope="chairs",
        include_armchairs=False,
        min_confidence=0.7,
        snap_step_deg=90,
        label_map={"C1": "chair_1"},
        yaw_history={"chair_1": [0]},
    )
    assert report["stop_reason"] == "blend_apply_failed"
    monkeypatch.setattr(repair, "_get_by_path", original_get_by_path)

    image = tmp_path / "view.png"
    image.write_bytes(b"png")
    bad_root = tmp_path / "bad_root.json"
    bad_root.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="Scene root"):
        repair.run_topview_vlm_orientation_repair(
            scene_path=bad_root,
            image_path=image,
            out_scene_path=tmp_path / "bad.out.json",
            out_review_path=tmp_path / "bad.review.json",
            out_report_path=tmp_path / "bad.report.json",
            provider="none",
            model=None,
            max_objects=10,
            target_scope="chairs",
            include_armchairs=False,
            min_confidence=0.7,
            max_delta_deg=180,
            snap_step_deg=90,
        )

    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(sample_scene()), encoding="utf-8")
    bad_label_map = tmp_path / "bad_labels.json"
    bad_label_map.write_text(json.dumps(["C1"]), encoding="utf-8")
    with pytest.raises(ValueError, match="target label map"):
        repair.run_topview_vlm_orientation_repair(
            scene_path=scene_path,
            image_path=image,
            out_scene_path=tmp_path / "labels.out.json",
            out_review_path=tmp_path / "labels.review.json",
            out_report_path=tmp_path / "labels.report.json",
            provider="none",
            model=None,
            max_objects=10,
            target_scope="chairs",
            include_armchairs=False,
            min_confidence=0.7,
            max_delta_deg=180,
            snap_step_deg=90,
            target_label_map_path=bad_label_map,
        )

    stale_label_map = tmp_path / "stale_labels.json"
    stale_label_map.write_text(json.dumps({"C9": "missing"}), encoding="utf-8")
    monkeypatch.setattr(
        repair,
        "call_openai_compatible_vlm",
        lambda **_kwargs: api_response(
            {"summary": "all apply", "objects": [{"label_id": "C1", "action": "set_yaw", "target_yaw_deg": 95, "confidence": 0.95}]}
        ),
    )
    report = repair.run_topview_vlm_orientation_repair(
        scene_path=scene_path,
        image_path=image,
        out_scene_path=tmp_path / "openai.out.json",
        out_review_path=tmp_path / "openai.review.json",
        out_report_path=tmp_path / "openai.report.json",
        provider="openai",
        model="mock",
        max_objects=10,
        target_scope="all",
        include_armchairs=True,
        min_confidence=0.7,
        max_delta_deg=180,
        snap_step_deg=90,
        target_label_map_path=stale_label_map,
    )
    assert report["provider"] == "openai"
    assert report["counts"]["applied"] == 1
    monkeypatch.setattr(
        repair,
        "call_ollama_vlm",
        lambda **_kwargs: api_response(
            {"summary": "ollama all", "objects": [{"label_id": "C1", "action": "keep", "confidence": 1.0}]}
        ),
    )
    report = repair.run_topview_vlm_orientation_repair(
        scene_path=scene_path,
        image_path=image,
        out_scene_path=tmp_path / "ollama_all.out.json",
        out_review_path=tmp_path / "ollama_all.review.json",
        out_report_path=tmp_path / "ollama_all.report.json",
        provider="ollama",
        model="mock",
        max_objects=10,
        target_scope="all",
        include_armchairs=True,
        min_confidence=0.7,
        max_delta_deg=180,
        snap_step_deg=90,
    )
    assert report["provider"] == "ollama"
    assert report["stop_reason"] == "converged_keep"


def test_topview_remaining_http_and_parser_edges(tmp_path, monkeypatch):
    assert repair._extract_position_xy({"cx": "1.5", "cy": "2.5"}) == (1.5, 2.5)
    assert repair._extract_size_xy({"transform": {"size": [0.4, 0.7]}}) == (0.4, 0.7)
    assert repair.collect_scene_objects({"not_objects": {"nested": []}}) == []
    refs = repair.collect_scene_objects({"objects": ["bad", {"id": "x", "category": "misc"}]})
    assert [ref.object_id for ref in refs] == ["x"]

    source_ref = repair.SceneObjectRef(
        "src",
        "misc",
        "office chair",
        (),
        {
            "source": {"blend_object_name": "OfficeChair"},
            "meta": {
                "affordance": "table_chair",
                "supplier_candidate": {"category_norm": "chair", "semantic_group": "chair", "title": "Office chair"},
            },
        },
        None,
        (0.0, 0.0),
        (0.5, 0.5),
    )
    assert repair.is_chair_like(source_ref)
    assert repair._semantic_blob(source_ref).count("chair") >= 2
    assert repair.is_table_like(repair.SceneObjectRef("d", "misc", "DeskFactory custom", (), {}, None, (0, 0), (1, 1)))
    assert repair.is_table_like(repair.SceneObjectRef("t", "misc", "coffee table", (), {}, None, (0, 0), (1, 1)))
    assert repair.is_table_like(repair.SceneObjectRef("ru", "misc", "письменный стол", (), {}, None, (0, 0), (1, 1)))

    original_get_by_path = repair._get_by_path
    monkeypatch.setattr(repair, "_get_by_path", lambda _root, _path: None)
    _updated, report = repair.set_scene_object_yaws(sample_scene(), {"chair_1": 45})
    assert report["skipped"][0]["reason"] == "path_does_not_resolve_to_object"
    monkeypatch.setattr(repair, "_get_by_path", original_get_by_path)

    review_path = tmp_path / "review_edge.json"
    review_path.write_text(json.dumps({"summary": "edge", "objects": ["bad", {"object_id": "", "action": "keep"}]}), encoding="utf-8")
    assert repair.load_review_from_file(review_path).decisions == []
    assert repair._snap_yaw(-15, 0) == 345.0
    assert repair._decision_target_yaw(
        repair.OrientationDecision("x", "x", "rotate_clockwise", 90, 123, 0.9, ""),
        None,
    ) == 123.0
    assert repair._decision_target_yaw(
        repair.OrientationDecision("x", "x", "custom", None, 222, 0.9, ""),
        0,
    ) == 222.0

    image = tmp_path / "view.jpg"
    image.write_bytes(b"jpg")
    for key in list(repair.os.environ):
        if key == "OPENROUTER_API_KEY" or key == "ivangrigin_OPENROUTER_API_KEY" or key.startswith("ivangrigin_OPENROUTER_API_KEY_"):
            monkeypatch.delenv(key, raising=False)
    repair._DOTENV_LOADED = True
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        repair._provider_config("openrouter", None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        repair.call_openai_compatible_vlm(provider="openrouter", model=None, prompt="p", image_path=image)

    monkeypatch.setenv("OPENROUTER_API_KEY", "k0")
    monkeypatch.setenv("ivangrigin_OPENROUTER_API_KEY_1", "k1")

    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self, code):
            super().__init__("http://example.test", code, "bad", {}, None)

        def read(self):
            return b"detail"

    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"summary\":\"ok\",\"objects\":[]}"}}]}).encode()

    def flaky_openrouter(_request, timeout=0):
        del timeout
        calls["count"] += 1
        if calls["count"] == 1:
            raise FakeHTTPError(429)
        return FakeResponse()

    monkeypatch.setattr(repair.urllib.request, "urlopen", flaky_openrouter)
    response = repair.call_openai_compatible_vlm(provider="openrouter", model=None, prompt="p", image_path=image)
    assert response["choices"][0]["message"]["content"].startswith("{")
    assert calls["count"] == 2

    def final_http_error(_request, timeout=0):
        del timeout
        raise FakeHTTPError(500)

    monkeypatch.setattr(repair.urllib.request, "urlopen", final_http_error)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        repair.call_openai_compatible_vlm(provider="openrouter", model=None, prompt="p", image_path=image)
    with pytest.raises(RuntimeError, match="Ollama VLM request failed: HTTP 500"):
        repair.call_ollama_vlm(model=None, prompt="p", image_path=image)
    with pytest.raises(RuntimeError, match="Ollama VLM request failed: HTTP 500"):
        repair.call_ollama_vlm_multi(model=None, prompt="p", image_paths=[image])

    monkeypatch.setattr(
        repair,
        "call_vlm_json",
        lambda **_kwargs: api_response({"summary": "selected", "objects": [{"label_id": "C1", "best_variant_id": "offset_000", "confidence": 0.9}]}),
    )
    variant_report = repair.run_topview_vlm_variant_selection(
        contact_sheet_path=image,
        label_map={"C1": "chair_1"},
        variants=[{"variant_id": "offset_000", "target_yaws_deg": {"chair_1": 0}}],
        out_prompt_path=tmp_path / "variant_prompt.txt",
        out_review_path=tmp_path / "variant_review.json",
        out_report_path=tmp_path / "variant_report.json",
        provider="openai",
        model="mock",
        min_confidence=0.5,
    )
    assert variant_report["stop_reason"] == "variant_selected"
