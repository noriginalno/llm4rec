"""Independent coordinator challenges, isolated from model-author work in progress.

Expected candidates, ranks, AP arithmetic and fallback counts are literal tiny
oracles. All model constructors are replaced with deterministic test doubles.
Most tests also replace build_fold with a fixed already-normalized fold; the
one source-zero/cold-item integration check uses only the public data contract.
Fault-injection failures describe coordinator defenses, not observed misconduct
by any actual model. No existing evidence output is modified.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sweettv import cli, pipeline
from sweettv.contracts import TrainingData
from sweettv.data import PreparedData, build_fold as real_build_fold


NAMES = ("popularity", "repeat", "cbf", "association", "mf", "ppr")
TEMPLATE = ["u2", "u0", "u1"]
CANDIDATE_INDICES = [1, 2, 3, 4, 5, 6]
CANDIDATE_IDS = [60, 10, 20, 30, 40, 50]
FALLBACK = [10, 20, 40, 30, 50]
CBF_PREDICTIONS = {"u2": [50, 10, 20, 40, 30], "u0": [60, 10, 20, 40, 30], "u1": FALLBACK}
PROTOCOL = {
    "synthetic": True,
    "development": {"train_end": 10, "forecast_start": 10, "forecast_end": 20},
    "primary": {"train_end": 20, "forecast_start": 20, "forecast_end": 30},
    "final": {"train_end": 30, "forecast_start": 30, "forecast_end": 40},
}


def _training_data():
    # Catalog 999 is globally most popular but is NOT a forecast candidate.
    # Candidate 60 is cold yet supported by CBF for u0; seen candidate 10 stays
    # eligible. Training-only user is not part of the template/evaluation cohort.
    return TrainingData(
        ["u0", "u1", "u2", "train-only"], np.array([999, 60, 10, 20, 30, 40, 50, 70]),
        csr_matrix([[100, 0, 4, 0, 0, 0, 0, 1], [0, 0, 0, 3, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 9]]),
        [{"tv_show_title": f"Program {item}"} for item in [999, 60, 10, 20, 30, 40, 50, 70]],
        {"train_end": 10, "training_only": True},
    )


def _labels():
    return {"u0": [10, 20, 30, 40, 50], "u1": [], "u2": [60, 10, 20, 30, 40],
            "train-only": [10, 20, 30, 40, 50]}


class _InjectedModelError(RuntimeError):
    pass


class _ScalarModel:
    def __init__(self, name, calls, fault=None):
        self.name, self.calls, self.fault = name, calls, fault
        self.diagnostics = {"test_double": True}

    def fit(self, data):
        self.data = data
        self.calls.append((self.name, "fit"))
        if self.fault == "fit_error":
            raise _InjectedModelError("deliberate fit failure")
        if self.fault == "fit_counts":
            data.interactions.data[0] += 1
        if self.fault == "fit_metadata":
            data.metadata[0]["tv_show_title"] = "mutated"
        return self

    def _vector(self, user_index, candidate_indices):
        raw = self.data.item_ids[candidate_indices]
        if self.name == "popularity":
            values = np.asarray(self.data.interactions.sum(axis=0)).ravel()[candidate_indices].astype(float)
            values[values <= 0] = np.nan
            return values
        if self.name == "repeat":
            values = self.data.interactions[int(user_index), candidate_indices].toarray().ravel().astype(float)
            values[values <= 0] = np.nan
            return values
        values = np.full(len(candidate_indices), np.nan)
        if self.name == "cbf":
            user = self.data.user_ids[int(user_index)]
            if user == "u0":
                values[raw == 60], values[raw == 10] = 7, 6
            elif user == "u2":
                values[raw == 50] = 9
        return values

    def score(self, user_index, candidate_indices):
        self.calls.append((self.name, "score", [int(user_index)], candidate_indices.copy()))
        return self._vector(user_index, candidate_indices)


class _BatchModel(_ScalarModel):
    def score_batch(self, user_indices, candidate_indices):
        self.calls.append((self.name, "score_batch", user_indices.copy(), candidate_indices.copy()))
        if self.fault == "score_error":
            raise _InjectedModelError("deliberate scoring failure")
        if self.fault == "candidate_mutation":
            candidate_indices[:] = candidate_indices[::-1]
        if self.fault == "user_mutation":
            user_indices[:] = user_indices[::-1]
        if self.fault == "score_counts":
            self.data.interactions.data[0] += 1
        values = np.vstack([self._vector(user, candidate_indices) for user in user_indices])
        if self.fault == "one_dimensional":
            return values[0]
        if self.fault == "transposed":
            return values.T
        if self.fault == "short_candidates":
            return values[:, :-1]
        if self.fault == "extra_users":
            return np.vstack([values, values[:1]])
        if self.fault == "positive_infinity":
            values[0, 0] = np.inf
        if self.fault == "negative_infinity":
            values[0, 0] = -np.inf
        return values


def _install(monkeypatch, *, fault=None, scalar=False, labels=None):
    calls = []
    prepared = PreparedData(TEMPLATE[:], [], [], {"fixture": "unverified", "schema_verified_on_actual_source": False})

    def fold(*args, **kwargs):
        data = _training_data()
        return data, np.array(CANDIDATE_INDICES), deepcopy(_labels() if labels is None else labels), {
            "provenance": deepcopy(data.provenance), "test_double": True,
        }

    def factory(name, parameters=None):
        cls = _ScalarModel if scalar else _BatchModel
        return cls(name, calls, fault=fault if name == "cbf" else None)

    monkeypatch.setattr(pipeline, "build_fold", fold)
    monkeypatch.setattr(pipeline, "create_model", factory)
    return prepared, calls


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("scalar,batch_size", [(False, 1), (False, 2), (False, 100), (True, 2)])
def test_common_cohort_candidates_csv_and_evidence_only_fallback(monkeypatch, tmp_path, scalar, batch_size):
    prepared, calls = _install(monkeypatch, scalar=scalar)
    real_evaluate = pipeline.evaluate_predictions
    cohorts = []

    def track(predictions, labels, users, k=5):
        cohorts.append(users[:])
        return real_evaluate(predictions, labels, users, k)

    monkeypatch.setattr(pipeline, "evaluate_predictions", track)
    output = tmp_path / "fold"
    summary = pipeline.run_fold(prepared, 10, 10, 20, output, synthetic=True, batch_size=batch_size,
                                fusions={"fusion": {"cbf": 1.0, "mf": 100.0}})
    expected_names = set(NAMES) | {"fusion"}
    assert set(summary["primary_map_at_5"]) == expected_names
    # u0 has three distinct TRAINING programs (including noncandidates), so the
    # sparse diagnostic cohort is u2,u1. No method-specific evidence filtering.
    assert cohorts == [["u2", "u0"], TEMPLATE, ["u2", "u1"]] * 7
    assert _json(output / "evaluation_users.json") == ["u2", "u0"]
    assert _json(output / "candidate_ids.json") == CANDIDATE_IDS
    for call in calls:
        if call[1] in ("score", "score_batch"):
            assert call[3].tolist() == CANDIDATE_INDICES
            assert len(call[2]) <= batch_size
    assert _json(output / "predictions_cbf.json") == CBF_PREDICTIONS
    # MF has NO finite evidence. Its 100x fusion weight must not turn its
    # popular fallback list into votes. CBF raw 60 outranks 10 for u0 with RRF
    # evidence 1/61 versus 1/62; all other positions backfill exactly once.
    assert _json(output / "predictions_fusion.json") == CBF_PREDICTIONS
    for name, positions, no_evidence in [("cbf", 12, 1), ("fusion", 12, 1), ("mf", 15, 3),
                                         ("repeat", 11, 0), ("popularity", 3, 0)]:
        assert summary["support"][name] == {
            "fallback_positions": positions, "fallback_users": 3, "no_evidence_users": no_evidence,
        }
    for name in expected_names:
        predictions = _json(output / f"predictions_{name}.json")
        with (output / f"submission_{name}.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows == [["user_id", "tv_show_id"], *[[user, " ".join(map(str, predictions[user]))] for user in TEMPLATE]]
        for items in predictions.values():
            assert len(set(items)) == 5 and set(items) <= set(CANDIDATE_IDS)
    metrics = _json(output / "metrics.json")["cbf"]
    # Both primary users hit ranks 2,3,4,5: (.5+2/3+.75+.8)/5 = 163/300.
    assert metrics["primary_five_label_cohort"]["map_at_5"] == pytest.approx(163 / 300)
    assert metrics["all_template_users_diagnostic"]["map_at_5"] == pytest.approx(163 / 450)
    with (output / "recommendations.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7 * 3 * 5
    cbf_sources = [row for row in rows if row["method"] == "cbf"]
    assert sum(row["source"] == "training_popularity_backfill" for row in cbf_sources) == 12
    assert all(row["score"] == "" for row in cbf_sources if row["source"] == "training_popularity_backfill")
    assert summary["example_users"] == ["u1", "u2"]
    assert summary["synthetic"] is True and summary["official_kaggle_data"] is False
    assert summary["kaggle_score"] is None and summary["submitted"] is False


@pytest.mark.parametrize("fault", ["one_dimensional", "transposed", "short_candidates", "extra_users", "positive_infinity", "negative_infinity"])
def test_batch_shape_or_infinity_is_an_error_not_a_silent_baseline(monkeypatch, tmp_path, fault):
    prepared, _ = _install(monkeypatch, fault=fault)
    output = tmp_path / "fold"
    with pytest.raises(RuntimeError, match="Invalid score shape/values from cbf"):
        pipeline.run_fold(prepared, 10, 10, 20, output, synthetic=True)
    assert not (output / "summary.json").exists()
    assert not list(output.glob("submission_*.csv"))


@pytest.mark.parametrize("fault", ["fit_error", "score_error"])
def test_model_exceptions_propagate_without_publishing_a_success_summary(monkeypatch, tmp_path, fault):
    prepared, _ = _install(monkeypatch, fault=fault)
    output = tmp_path / "fold"
    with pytest.raises(_InjectedModelError):
        pipeline.run_fold(prepared, 10, 10, 20, output, synthetic=True)
    assert not (output / "summary.json").exists()
    assert not list(output.glob("submission_*.csv"))


@pytest.mark.parametrize("fault", ["fit_counts", "fit_metadata", "score_counts"])
def test_shared_count_and_metadata_mutations_are_detected(monkeypatch, tmp_path, fault):
    prepared, _ = _install(monkeypatch, fault=fault)
    with pytest.raises(RuntimeError, match="mutated shared training data"):
        pipeline.run_fold(prepared, 10, 10, 20, tmp_path / "fold", synthetic=True)


@pytest.mark.parametrize("fault", ["candidate_mutation", "user_mutation"])
def test_mutable_query_arrays_cannot_silently_change_candidate_or_user_alignment(monkeypatch, tmp_path, fault):
    prepared, _ = _install(monkeypatch, fault=fault)
    output = tmp_path / "fold"
    try:
        pipeline.run_fold(prepared, 10, 10, 20, output, synthetic=True)
    except (RuntimeError, ValueError) as error:
        assert any(word in str(error).lower() for word in ("mutat", "read-only", "candidate", "query"))
        return
    actual = _json(output / "predictions_cbf.json")
    popularity = _json(output / "predictions_popularity.json")
    pytest.fail(f"Accepted {fault}; CBF u2={actual['u2']}, expected {CBF_PREDICTIONS['u2']}; "
                f"popularity u2={popularity['u2']}, expected {FALLBACK}. No coordinator mutation error.")


@pytest.mark.parametrize("field", ["provenance", "matrix_shape"])
def test_training_fingerprint_covers_provenance_and_sparse_matrix_shape(field):
    data = _training_data()
    before = pipeline.training_fingerprint(data)
    if field == "provenance":
        data.provenance["train_end"] = 999
    else:
        # A faulty model can leave the same CSR buffers but invalidate the
        # users/items shape contract. The advertised mutation guard should see it.
        data.interactions.resize((4, 9))
    assert pipeline.training_fingerprint(data) != before, f"Fingerprint missed {field} mutation"


def test_unverified_input_is_not_positively_claimed_as_official_kaggle_data(monkeypatch, tmp_path):
    prepared, _ = _install(monkeypatch)
    # No official download/source provenance exists. Default synthetic=False
    # alone cannot establish that these handmade observations came from Kaggle.
    summary = pipeline.run_fold(prepared, 10, 10, 20, tmp_path / "fold")
    assert summary["official_kaggle_data"] is not True, "Absence of a synthetic flag is not verified source provenance"


@pytest.mark.parametrize("operation", ["fold", "full_run", "save_json"])
def test_existing_output_evidence_is_never_overwritten(monkeypatch, tmp_path, operation):
    prepared, calls = _install(monkeypatch)
    path = tmp_path / "existing"
    if operation == "save_json":
        path.write_bytes(b"existing artifact\n")
        sentinel = path
    else:
        path.mkdir()
        sentinel = path / "keep.txt"
        sentinel.write_bytes(b"existing artifact\n")
    with pytest.raises(FileExistsError):
        if operation == "fold":
            pipeline.run_fold(prepared, 10, 10, 20, path, synthetic=True)
        elif operation == "full_run":
            cli.full_run(prepared, path, PROTOCOL, synthetic=True)
        else:
            pipeline.save_json(path, {"replacement": True})
    assert sentinel.read_bytes() == b"existing artifact\n"
    assert calls == [], "Existing-output guard should run before model fitting"


def test_two_example_users_are_training_only_middle_activity_extremes():
    data = TrainingData(["a", "b", "c", "d", "e", "f", "cold"], np.arange(1, 5),
                        csr_matrix([[1, 0, 0, 0], [2, 2, 0, 0], [6, 0, 0, 0], [6, 2, 0, 0],
                                    [4, 3, 3, 0], [50, 0, 0, 0], [0, 0, 0, 0]]), [{} for _ in range(4)])
    # Sorted activity a,b,c,d,e,f -> middle pool b,c,d -> shares .5,1,.75.
    expected = ["b", "c"]
    for template in (["cold", "f", "e", "d", "c", "b", "a"], data.user_ids):
        chosen, details = pipeline.select_example_users(data, template)
        assert chosen == expected
        assert set(details["users"]) == set(expected)


def test_future_labels_do_not_change_example_selection_or_predictions(monkeypatch, tmp_path):
    prepared, _ = _install(monkeypatch)
    first = pipeline.run_fold(prepared, 10, 10, 20, tmp_path / "first", synthetic=True)
    changed = {user: [] for user in _labels()}
    prepared, _ = _install(monkeypatch, labels=changed)
    second = pipeline.run_fold(prepared, 10, 10, 20, tmp_path / "second", synthetic=True)
    assert first["example_users"] == second["example_users"] == ["u1", "u2"]
    assert first["training_fingerprint"] == second["training_fingerprint"]
    assert second["primary_user_count"] == 0
    assert all(value is None for value in second["primary_map_at_5"].values())
    for name in NAMES:
        assert _json(tmp_path / "first" / f"predictions_{name}.json") == _json(tmp_path / "second" / f"predictions_{name}.json")


def _install_full_run(monkeypatch, *, empty_stage=None, collision=False):
    calls = []

    def fold(prepared, train_end, forecast_start, forecast_end, output, **kwargs):
        calls.append({"stage": output.name, "bounds": (train_end, forecast_start, forecast_end), **deepcopy(kwargs)})
        output.mkdir()
        if output.name == "development":
            values = {"repeat": 1.0, "cbf": .1, "hybrid_repeat": .2, "hybrid_balanced": .9}
        elif output.name == "primary":
            values = {"repeat": .2, "cbf": .8, "hybrid_balanced": .6}
        else:
            values = {}
            with (output / "submission_cbf.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["user_id", "tv_show_id"])
                writer.writerows((user, "60 50 40 30 20") for user in prepared.template_users)
            (output / "candidate_ids.json").write_text(json.dumps([10, 20, 30, 40, 50, 60]), encoding="utf-8")
            if collision:
                (output / "submission.csv").write_bytes(b"existing final artifact\n")
        return {"primary_map_at_5": {name: None for name in values} if output.name == empty_stage else values}

    monkeypatch.setattr(cli, "run_fold", fold)
    return calls


def test_development_freezes_fusion_but_primary_selects_final_method_and_exact_csv(monkeypatch, tmp_path):
    calls = _install_full_run(monkeypatch)
    prepared = PreparedData(TEMPLATE[:], [], [], {"fixture": True})
    parameters = {"mf": {"factors": 4}}
    result = cli.full_run(prepared, tmp_path / "run", PROTOCOL, synthetic=True, parameters=parameters)
    assert [call["stage"] for call in calls] == ["development", "primary", "final"]
    assert calls[0]["fusions"] == pipeline.DEFAULT_FUSIONS
    frozen = {"hybrid_balanced": pipeline.DEFAULT_FUSIONS["hybrid_balanced"]}
    assert calls[1]["fusions"] == calls[2]["fusions"] == frozen
    assert all(call["model_parameters"] == parameters for call in calls)
    assert calls[2]["evaluate"] is False
    assert result["selected_method"] == "cbf"  # NOT development's repeat winner.
    source = tmp_path / "run" / "final" / "submission_cbf.csv"
    final = source.with_name("submission.csv")
    assert source.read_bytes() == final.read_bytes()
    assert result["validation"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["kaggle_score"] is None and result["submitted"] is False


@pytest.mark.parametrize("stage,stages_called", [("development", ["development"]), ("primary", ["development", "primary"])])
def test_full_run_with_no_primary_evaluation_users_abstains_without_final_fit(monkeypatch, tmp_path, stage, stages_called):
    calls = _install_full_run(monkeypatch, empty_stage=stage)
    prepared = PreparedData(TEMPLATE[:], [], [], {})
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="nonempty primary evaluation cohort"):
        cli.full_run(prepared, output, PROTOCOL, synthetic=True)
    assert [call["stage"] for call in calls] == stages_called
    assert not (output / "final").exists()
    assert not (output / "selection.json").exists()


def test_final_selected_copy_does_not_overwrite_a_destination_created_during_final_fit(monkeypatch, tmp_path):
    _install_full_run(monkeypatch, collision=True)
    prepared = PreparedData(TEMPLATE[:], [], [], {})
    output = tmp_path / "run"
    try:
        cli.full_run(prepared, output, PROTOCOL, synthetic=True)
    except FileExistsError:
        assert (output / "final" / "submission.csv").read_bytes() == b"existing final artifact\n"
        return
    assert (output / "final" / "submission.csv").read_bytes() == b"existing final artifact\n", "Final copy silently overwrote an existing destination"
    pytest.fail("A destination collision must abort selection instead of publishing success")


def test_protocol_confirmation_is_checked_before_expensive_raw_loading(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(cli, "load", lambda args: loaded.append(args.raw_dir) or PreparedData(["u"], [], [], {}))
    monkeypatch.setattr(sys, "argv", ["sweettv", "run", "--raw-dir", str(tmp_path / "raw"),
                                     "--protocol", str(tmp_path / "not-read.json"), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert loaded == [], "Unconfirmed run invoked full load_raw/reconstruction before rejecting the command"


def test_confirmed_cli_run_preserves_protocol_and_synthetic_marker(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "SYNTHETIC.json").write_text("{}", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(PROTOCOL), encoding="utf-8")
    calls = []
    prepared = PreparedData(["u"], [], [], {})
    monkeypatch.setattr(cli, "load", lambda args: prepared)
    monkeypatch.setattr(cli, "full_run", lambda data, out, spec, **kw: calls.append((data, out, spec, kw)) or {"synthetic": kw["synthetic"]})
    monkeypatch.setattr(sys, "argv", ["sweettv", "run", "--raw-dir", str(raw), "--protocol", str(protocol),
                                     "--confirm-protocol", "--out", str(tmp_path / "out")])
    cli.main()
    assert len(calls) == 1 and calls[0][0] is prepared
    assert calls[0][2] == PROTOCOL
    assert calls[0][3] == {"synthetic": True, "parameters": None}
    assert json.loads(capsys.readouterr().out)["synthetic"] is True


def test_source_schedule_zero_and_missing_ids_never_replace_cold_candidates(monkeypatch, tmp_path):
    prepared, calls = _install(monkeypatch)
    schedules = [{"airing_id": "train", "channel_id": "7", "start": 0, "end": 10, "tv_show_id": 10, "metadata": {}}]
    for item in [0, None, 10, 20, 30, 40, 50, 60]:
        schedules.append({"airing_id": f"forecast-{item}", "channel_id": f"c-{item}", "start": 10,
                          "end": 20, "tv_show_id": item, "metadata": {}})
    events = [{"user_id": "u", "tv_show_id": 10, "start": 0, "end": 10, "airing_id": "train"}]
    events += [{"user_id": "u", "tv_show_id": item, "start": 10, "end": 20, "airing_id": f"forecast-{item}"}
               for item in [10, 20, 30, 40, 50]]
    prepared = PreparedData(["u"], schedules, events, {"fixture": True})
    monkeypatch.setattr(pipeline, "build_fold", real_build_fold)
    output = tmp_path / "fold"
    summary = pipeline.run_fold(prepared, 10, 10, 20, output, synthetic=True)
    assert _json(output / "candidate_ids.json") == [10, 20, 30, 40, 50, 60]
    assert summary["split"]["seen_candidate_items"] == 1
    assert summary["split"]["cold_candidate_items"] == 5
    assert _json(output / "predictions_cbf.json") == {"u": [10, 20, 30, 40, 50]}
    assert summary["primary_map_at_5"]["cbf"] == 1.0
