"""Reading back which metric a run's off-grid checkpoints followed.

The cases that matter are the two the audit exists to tell apart -- a run saving on
the judged metric and a run saving on the overall loss -- plus the case where the
two series pick the same steps and the comparison proves nothing.

Since 2026-11-10 the outcome is two columns out of a fixed vocabulary rather than
one column of Japanese sentences, and the output claims the report TSV convention
in its name and its first line. Both are tested here: a column whose values are
prose cannot be counted, and a file whose name does not say what shape it is has
to be opened to find out.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def repo_root() -> Path:
    """The checkout this test file belongs to, not git's common dir."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    path = repo_root() / "scripts" / "audit_save_on_improve_metric.py"
    spec = importlib.util.spec_from_file_location("audit_save_on_improve_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history(mask_values, loss_values, step=100):
    """One log_history with the two series at the same steps."""
    out = []
    for i, (m, loss) in enumerate(zip(mask_values, loss_values), start=1):
        s = i * step
        out.append({"step": s, "eval_loss": loss})
        out.append({"step": s, "eval_loss_mask": m})
    return out


def write_run(root: Path, name: str, log_history, ckpt_steps, best=None):
    run = root / name
    for s in ckpt_steps:
        (run / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)
    state = {"global_step": max(ckpt_steps), "log_history": log_history}
    if best is not None:
        state["best_metric"] = best[0]
        state["best_model_checkpoint"] = f"/somewhere/{name}/checkpoint-{best[1]}"
    (run / f"checkpoint-{max(ckpt_steps)}" / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8")
    return run


def read_tsv(path: Path):
    """The data rows. The provenance block is `#` lines above the header."""
    with open(path, newline="", encoding="utf-8") as fh:
        body = [l for l in fh.read().split("\n") if l and not l.startswith("#")]
    return list(csv.DictReader(body, delimiter="\t"))


def test_improvement_steps_restart_at_the_given_step(mod):
    points = {100: 0.5, 200: 0.4, 300: 0.6, 400: 0.45}
    assert mod.improvement_steps(points) == {100, 200}
    # From 300 the first point seen becomes the best, so 400 is an improvement.
    assert mod.improvement_steps(points, start=300) == {300, 400}


def test_a_run_following_the_judged_metric_is_named(mod, tmp_path):
    # mask improves at 300; overall loss improves at 400 instead.
    mask = [0.9, 0.8, 0.7, 0.75, 0.76]
    loss = [0.9, 0.8, 0.85, 0.7, 0.72]
    run = write_run(tmp_path, "arm", history(mask, loss), [300, 500])
    mod.main(["--run", str(run), "--save-steps", "500", "--out", str(tmp_path / "rt1-o.tsv")])
    row = read_tsv(tmp_path / "rt1-o.tsv")[0]
    assert row["off_grid"] == "1"
    assert row["on_judged"] == "1/1" and row["on_other"] == "0/1"
    assert row["outcome"] == mod.FOLLOWED
    assert row["metric_followed"] == "eval_loss_mask"


def test_a_run_following_the_overall_loss_is_named(mod, tmp_path):
    mask = [0.9, 0.8, 0.7, 0.75, 0.76]
    loss = [0.9, 0.8, 0.85, 0.7, 0.72]
    run = write_run(tmp_path, "arm", history(mask, loss), [400, 500])
    mod.main(["--run", str(run), "--save-steps", "500", "--out", str(tmp_path / "rt1-o.tsv")])
    row = read_tsv(tmp_path / "rt1-o.tsv")[0]
    assert row["on_judged"] == "0/1" and row["on_other"] == "1/1"
    assert row["outcome"] == mod.FOLLOWED
    assert row["metric_followed"] == "eval_loss"


def test_two_series_that_agree_are_reported_as_undecidable(mod, tmp_path):
    same = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(same, list(same)), [300, 400])
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(tmp_path / "rt1-o.tsv")])
    row = read_tsv(tmp_path / "rt1-o.tsv")[0]
    assert row["outcome"] == mod.SERIES_AGREE
    assert row["metric_followed"] == ""
    assert row["on_judged"] == "1/1" and row["on_other"] == "1/1"


def test_a_run_with_no_off_grid_checkpoint_says_so(mod, tmp_path):
    mask = [0.9, 0.8, 0.7, 0.6]
    run = write_run(tmp_path, "arm", history(mask, mask), [200, 400])
    mod.main(["--run", str(run), "--save-steps", "200", "--out", str(tmp_path / "rt1-o.tsv")])
    assert read_tsv(tmp_path / "rt1-o.tsv")[0]["outcome"] == mod.NO_OFF_GRID


def test_the_best_checkpoint_is_read_as_a_step_not_a_path(mod, tmp_path):
    """The field holds an absolute path from train time; the tree may be gone."""
    mask = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(mask, mask), [300, 400],
                    best=(0.7, 300))
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(tmp_path / "rt1-o.tsv")])
    row = read_tsv(tmp_path / "rt1-o.tsv")[0]
    assert row["best_checkpoint_step"] == "300"
    assert row["best_metric_recorded"] == "0.700000"
    assert row["judged_min_step"] == "300" and row["judged_min"] == "0.700000"


def test_the_best_restarts_where_the_stage_resumed(mod, tmp_path):
    """A continuation has no memory of the previous stage's minimum.

    The restored history sits lower than anything this stage reaches, so counting
    from the first step in the file finds no improvement at all. The resume step is
    given with @STEP because the two portions are one ascending series on disk.
    """
    restored = history([0.5, 0.5], [0.5, 0.5])                      # steps 100, 200
    own = [{"step": 300, "eval_loss": 0.9}, {"step": 300, "eval_loss_mask": 0.9},
           {"step": 400, "eval_loss": 0.8}, {"step": 400, "eval_loss_mask": 0.7}]
    run = write_run(tmp_path, "arm", restored + own, [400, 500])
    out = tmp_path / "rt1-o.tsv"
    mod.main(["--run", f"{run}@200", "--save-steps", "500", "--out", str(out)])
    row = read_tsv(out)[0]
    assert row["resumed_at"] == "200"
    assert row["off_grid"] == "1"
    assert row["on_judged"] == "1/1"


def test_without_the_resume_step_the_same_run_finds_nothing(mod, tmp_path):
    """The failure the @STEP argument exists to prevent, kept as a test."""
    restored = history([0.5, 0.5], [0.5, 0.5])
    own = [{"step": 300, "eval_loss": 0.9}, {"step": 300, "eval_loss_mask": 0.9},
           {"step": 400, "eval_loss": 0.8}, {"step": 400, "eval_loss_mask": 0.7}]
    run = write_run(tmp_path, "arm", restored + own, [400, 500])
    out = tmp_path / "rt1-o.tsv"
    mod.main(["--run", str(run), "--save-steps", "500", "--out", str(out)])
    assert read_tsv(out)[0]["on_judged"] == "0/1"


def test_a_checkpoint_from_before_the_resume_is_not_counted(mod, tmp_path):
    """Those were written by the stage before this one."""
    restored = history([0.9, 0.8], [0.9, 0.8])                      # steps 100, 200
    own = [{"step": 300, "eval_loss": 0.7}, {"step": 300, "eval_loss_mask": 0.7}]
    run = write_run(tmp_path, "arm", restored + own, [100, 300])
    out = tmp_path / "rt1-o.tsv"
    mod.main(["--run", f"{run}@200", "--save-steps", "300", "--out", str(out)])
    row = read_tsv(out)[0]
    assert row["checkpoints"] == "2"
    assert row["off_grid"] == "0"


def test_a_run_with_no_checkpoint_stops(mod, tmp_path):
    (tmp_path / "arm").mkdir()
    with pytest.raises(SystemExit):
        mod.main(["--run", str(tmp_path / "arm"), "--out", str(tmp_path / "rt1-o.tsv")])


def test_the_columns_are_stable(mod):
    assert mod.COLUMNS[:5] == ["run", "resumed_at", "save_steps", "checkpoints",
                               "off_grid"]
    assert "outcome" in mod.COLUMNS and "on_judged" in mod.COLUMNS


def test_the_count_of_evaluation_points_is_one_column_named_for_what_it_holds(mod, tmp_path):
    """It used to be two columns called judged_metric and other_metric, which said
    they held a metric's name and held a count of evaluation points instead."""
    mask = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(mask, mask), [300, 400])
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(tmp_path / "rt1-o.tsv")])
    row = read_tsv(tmp_path / "rt1-o.tsv")[0]
    assert row["eval_points"] == "4"
    assert "judged_metric" not in row and "other_metric" not in row


def test_a_run_whose_series_have_different_lengths_is_said_out_loud(mod, tmp_path, capsys):
    """The single count column stands for both series, so when it cannot, say so."""
    log = history([0.9, 0.8, 0.7, 0.75], [0.9, 0.8, 0.7, 0.75])
    log = [e for e in log if not (e.get("step") == 400 and "eval_loss" in e)]
    run = write_run(tmp_path, "arm", log, [300, 400])
    out = tmp_path / "rt1-o.tsv"
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(out)])
    assert "2 系列の評価点数が違う" in capsys.readouterr().out
    assert "logged at a different number of points" in out.read_text(encoding="utf-8")


def test_every_outcome_written_is_from_the_fixed_vocabulary(mod, tmp_path):
    mask = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(mask, mask), [300, 400])
    out = tmp_path / "rt1-o.tsv"
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(out)])
    assert all(r["outcome"] in mod.OUTCOMES for r in read_tsv(out))


def test_the_name_has_to_claim_the_convention(mod, tmp_path):
    mask = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(mask, mask), [300, 400])
    with pytest.raises(SystemExit):
        mod.main(["--run", str(run), "--save-steps", "400",
                  "--out", str(tmp_path / "plain.tsv")])
    assert not (tmp_path / "plain.tsv").exists()


def test_the_provenance_block_comes_before_the_header(mod, tmp_path):
    mask = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(mask, mask), [300, 400])
    out = tmp_path / "rt1-o.tsv"
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(out),
              "--model", "compounds bert small"])
    lines = out.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "#! report-tsv 1"
    assert "# model: compounds bert small" in lines
    header = next(l for l in lines if not l.startswith("#"))
    assert header.split("\t") == mod.COLUMNS
    assert b"\r" not in out.read_bytes()


def test_a_spec_without_an_at_sign_resumes_at_zero(mod, tmp_path):
    assert mod.split_run("/a/b") == (Path("/a/b"), 0)
    assert mod.split_run("/a/b@30000") == (Path("/a/b"), 30000)
