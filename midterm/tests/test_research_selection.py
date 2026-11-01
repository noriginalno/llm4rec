import json

import pytest

from tools import select_research_shortlist as selection


def write_trial(root, name, scores):
    for fold, (strict, all_user) in zip(selection.FOLDS, scores):
        folder = root / fold
        (folder / name).mkdir(parents=True)
        for filename, value in (("labels.json", {"u": [1, 2, 3, 4, 5]}),
                                ("candidate_ids.json", [1, 2, 3, 4, 5]),
                                ("template_users.json", ["u"])):
            (folder / filename).write_text(json.dumps(value))
        (folder / name / "result.json").write_text(json.dumps({
            "map_at_5": strict, "all_user_map_at_5": all_user}))


def run_selection(tmp_path, monkeypatch, trials):
    raw, fusion, output = tmp_path / "raw", tmp_path / "fusion", tmp_path / "selected"
    write_trial(raw, selection.RAW_NAME, [(0.4, 0.3), (0.4, 0.3)])
    specs = []
    for name, scores in trials:
        write_trial(fusion, name, scores)
        specs.append({"name": name, "class": "research_v2.fusion:EvidenceBlendModel",
                      "parameters": {"components": [{
                          "name": "legacy", "class": "research_v2.legacy:LegacyHybrid", "weight": 1}]}})
    (fusion / "grid.json").write_text(json.dumps(specs))
    monkeypatch.setattr(selection, "RAW_ROOT", raw)
    monkeypatch.setattr("sys.argv", ["select_research_shortlist.py", "--fusion-root", str(fusion),
                                     "--out", str(output)])
    selection.main()
    return json.loads((output / "selection.json").read_text()), json.loads((output / "grid.json").read_text())


@pytest.mark.parametrize("scores", [
    [(0.4, 0.3), (0.4, 0.3)],  # Equal is not incremental evidence.
    [(0.5, 0.4), (0.39, 0.31)],  # One-window win is insufficient.
    [(0.5, 0.29), (0.5, 0.4)],  # Do not hide an all-user loss.
])
def test_inconsistent_blends_do_not_enter_confirmation(tmp_path, monkeypatch, scores):
    result, grid = run_selection(tmp_path, monkeypatch, [("raw_repeat_w05", scores)])
    assert result["chosen_blend"] is None
    assert result["confirmation_new_candidates"] == [selection.RAW_NAME]
    assert len(grid) == 3  # Two controls, one new candidate.
    assert result["confirmation_seen"] is False
    assert result["new_kaggle_submission_authorized"] is False


def test_stable_plateau_prefers_simple_repeat_and_freezes_nested_tie_control(tmp_path, monkeypatch):
    result, grid = run_selection(tmp_path, monkeypatch, [
        ("raw_legacy_w05", [(0.45, 0.35), (0.45, 0.35)]),
        ("raw_repeat_w05", [(0.45, 0.3495), (0.45, 0.3495)]),
    ])
    assert result["chosen_blend"] == "raw_repeat_w05"
    assert len(result["confirmation_new_candidates"]) == 2
    assert len(grid) == 5
    assert grid[-1]["parameters"]["reverse_score_ties"] is True
    assert grid[-1]["parameters"]["components"][0]["parameters"]["reverse_score_ties"] is True
