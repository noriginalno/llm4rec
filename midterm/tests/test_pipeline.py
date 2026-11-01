"""Integration tests for coordinator-owned code, not recommendation-quality claims."""

import json
from pathlib import Path
import subprocess
import sys
import runpy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sweettv.contracts import TrainingData
from sweettv.data import RawSchema, load_raw
from sweettv.demo import make_demo
from sweettv.evaluation import reciprocal_rank_fusion
from sweettv.pipeline import MODEL_NAMES, best_method, run_fold, training_fingerprint
from sweettv.cli import full_run, week
from sweettv import pipeline


def test_fusion_uses_only_evidence_and_preserves_missing():
    scores = {"repeat": np.array([10.0, np.nan, np.nan, np.nan, np.nan, np.nan]),
              "cbf": np.array([np.nan, 0.9, np.nan, np.nan, np.nan, np.nan])}
    result = reciprocal_rank_fusion(scores, {"repeat": 1, "cbf": 0.5}, np.arange(6), np.arange(1, 7))
    assert result[0] == pytest.approx(1 / 61)
    assert result[1] == pytest.approx(0.5 / 61)
    assert np.isnan(result[2:]).all()


def test_fusion_repeat_only_bound_and_discovery_capable_bound():
    # Independent arithmetic from the adversarial design debate.
    ids = np.arange(1, 7)
    inputs = {"repeat": np.array([5, 4, 3, 2, 1, np.nan]),
              "cbf": np.array([np.nan] * 5 + [1]),
              "rules": np.array([np.nan] * 5 + [1])}
    low = reciprocal_rank_fusion(inputs, {"repeat": 1, "cbf": 0.2, "rules": 0.2}, np.arange(6), ids)
    high = reciprocal_rank_fusion(inputs, {"repeat": 1, "cbf": 0.5, "rules": 0.5}, np.arange(6), ids)
    assert low[5] == pytest.approx(0.4 / 61)
    assert low[5] < low[4]
    assert high[5] > high[4]


def test_empty_primary_cohort_cannot_select_winner():
    with pytest.raises(ValueError, match="nonempty"):
        best_method({"primary_map_at_5": {"repeat": None, "cbf": None}})


def test_exact_metric_tie_prefers_repeat_not_complexity():
    assert best_method({"primary_map_at_5": {"cbf": 0.5, "repeat": 0.5, "ppr": 0.4}}) == "repeat"


def test_training_fingerprint_covers_counts_ids_and_metadata():
    data = TrainingData(["u"], np.array([10, 20]), csr_matrix([[1, 0]]), [{"tv_show_title": "A"}, {}])
    first = training_fingerprint(data)
    data.metadata[0]["tv_show_title"] = "B"
    assert training_fingerprint(data) != first


def test_all_models_end_to_end_on_explicit_synthetic_raw_csv(tmp_path):
    raw = tmp_path / "raw"
    make_demo(raw, user_count=6)
    prepared = load_raw(raw, schema=RawSchema(log_start="start_time", log_end="end_time", schedule_start="start_time"))
    result = run_fold(prepared, week(2020, 19), week(2020, 19), week(2020, 31), tmp_path / "output", synthetic=True)
    assert result["synthetic"] is True
    assert result["kaggle_score"] is None and result["submitted"] is False
    assert set(result["primary_map_at_5"]) == set(MODEL_NAMES)
    assert all(row["valid"] and row["rows"] == 6 for row in result["submissions"].values())
    assert len(result["example_users"]) == 2
    before = (tmp_path / "output" / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_fold(prepared, week(2020, 19), week(2020, 19), week(2020, 31), tmp_path / "output", synthetic=True)
    assert (tmp_path / "output" / "summary.json").read_bytes() == before
    assert json.loads(before)["synthetic"] is True


def test_whole_run_independent_oracle_handles_json_candidate_list(tmp_path):
    raw = tmp_path / "raw"
    make_demo(raw, user_count=6)
    prepared = load_raw(raw, schema=RawSchema(log_start="start_time", log_end="end_time", schedule_start="start_time"))
    protocol = {
        "development": {"train_end": week(2020, 15), "forecast_start": week(2020, 15), "forecast_end": week(2020, 19)},
        "primary": {"train_end": week(2020, 19), "forecast_start": week(2020, 19), "forecast_end": week(2020, 31)},
        "final": {"train_end": week(2020, 31), "forecast_start": week(2020, 31), "forecast_end": week(2020, 43)},
    }
    out = tmp_path / "run"
    full_run(prepared, out, protocol, synthetic=True)
    script = Path(__file__).resolve().parents[1] / "tools" / "verify_run.py"
    process = subprocess.run([sys.executable, "-I", "-S", str(script), "--run", str(out),
                              "--template", str(raw / "submission.csv")], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["synthetic"] is True
    assert result["per_model_pass"] == 22
    assert result["selected_file_identity"] is True


def test_scalar_model_reusable_return_buffer_is_copied_before_next_query(monkeypatch, tmp_path):
    fixtures = runpy.run_path(str(Path(__file__).with_name("test_pipeline_adversarial.py")))
    prepared, calls = fixtures["_install"](monkeypatch)
    factory = pipeline.create_model

    class ReusedScalar(fixtures["_ScalarModel"]):
        def score(self, user, candidates):
            value = super().score(user, candidates)
            if not hasattr(self, "buffer"):
                self.buffer = value.copy()
            else:
                self.buffer[:] = value
            return self.buffer

    def with_reused_buffer(name, parameters=None):
        return ReusedScalar(name, calls) if name == "cbf" else factory(name, parameters)

    monkeypatch.setattr(pipeline, "create_model", with_reused_buffer)
    output = tmp_path / "fold"
    summary = pipeline.run_fold(prepared, 10, 10, 20, output, synthetic=True)
    assert json.loads((output / "predictions_cbf.json").read_text()) == {
        "u2": [50, 10, 20, 40, 30], "u0": [60, 10, 20, 40, 30], "u1": [10, 20, 40, 30, 50],
    }
    assert summary["support"]["cbf"]["fallback_positions"] == 12
    assert summary["support"]["cbf"]["no_evidence_users"] == 1
