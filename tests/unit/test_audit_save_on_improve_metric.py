"""Reading back which metric a run's off-grid checkpoints followed.

The cases that matter are the two the audit exists to tell apart -- a run saving on
the judged metric and a run saving on the overall loss -- plus the case where the
two series pick the same steps and the comparison proves nothing.
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
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


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
    mod.main(["--run", str(run), "--save-steps", "500", "--out", str(tmp_path / "o.tsv")])
    row = read_tsv(tmp_path / "o.tsv")[0]
    assert row["off_grid"] == "1"
    assert row["on_judged"] == "1/1" and row["on_other"] == "0/1"
    assert row["verdict"] == "eval_loss_mask"


def test_a_run_following_the_overall_loss_is_named(mod, tmp_path):
    mask = [0.9, 0.8, 0.7, 0.75, 0.76]
    loss = [0.9, 0.8, 0.85, 0.7, 0.72]
    run = write_run(tmp_path, "arm", history(mask, loss), [400, 500])
    mod.main(["--run", str(run), "--save-steps", "500", "--out", str(tmp_path / "o.tsv")])
    row = read_tsv(tmp_path / "o.tsv")[0]
    assert row["on_judged"] == "0/1" and row["on_other"] == "1/1"
    assert row["verdict"] == "eval_loss"


def test_two_series_that_agree_are_reported_as_undecidable(mod, tmp_path):
    same = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(same, list(same)), [300, 400])
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(tmp_path / "o.tsv")])
    row = read_tsv(tmp_path / "o.tsv")[0]
    assert "判定不能" in row["verdict"]
    assert row["on_judged"] == "1/1" and row["on_other"] == "1/1"


def test_a_run_with_no_off_grid_checkpoint_says_so(mod, tmp_path):
    mask = [0.9, 0.8, 0.7, 0.6]
    run = write_run(tmp_path, "arm", history(mask, mask), [200, 400])
    mod.main(["--run", str(run), "--save-steps", "200", "--out", str(tmp_path / "o.tsv")])
    assert read_tsv(tmp_path / "o.tsv")[0]["verdict"] == "追加保存なし"


def test_the_best_checkpoint_is_read_as_a_step_not_a_path(mod, tmp_path):
    """The field holds an absolute path from train time; the tree may be gone."""
    mask = [0.9, 0.8, 0.7, 0.75]
    run = write_run(tmp_path, "arm", history(mask, mask), [300, 400],
                    best=(0.7, 300))
    mod.main(["--run", str(run), "--save-steps", "400", "--out", str(tmp_path / "o.tsv")])
    row = read_tsv(tmp_path / "o.tsv")[0]
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
    out = tmp_path / "o.tsv"
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
    out = tmp_path / "o.tsv"
    mod.main(["--run", str(run), "--save-steps", "500", "--out", str(out)])
    assert read_tsv(out)[0]["on_judged"] == "0/1"


def test_a_checkpoint_from_before_the_resume_is_not_counted(mod, tmp_path):
    """Those were written by the stage before this one."""
    restored = history([0.9, 0.8], [0.9, 0.8])                      # steps 100, 200
    own = [{"step": 300, "eval_loss": 0.7}, {"step": 300, "eval_loss_mask": 0.7}]
    run = write_run(tmp_path, "arm", restored + own, [100, 300])
    out = tmp_path / "o.tsv"
    mod.main(["--run", f"{run}@200", "--save-steps", "300", "--out", str(out)])
    row = read_tsv(out)[0]
    assert row["checkpoints"] == "2"
    assert row["off_grid"] == "0"


def test_a_run_with_no_checkpoint_stops(mod, tmp_path):
    (tmp_path / "arm").mkdir()
    with pytest.raises(SystemExit):
        mod.main(["--run", str(tmp_path / "arm"), "--out", str(tmp_path / "o.tsv")])


def test_the_columns_are_stable(mod):
    assert mod.COLUMNS[:5] == ["run", "resumed_at", "save_steps", "checkpoints",
                               "off_grid"]
    assert "verdict" in mod.COLUMNS and "on_judged" in mod.COLUMNS


def test_a_spec_without_an_at_sign_resumes_at_zero(mod, tmp_path):
    assert mod.split_run("/a/b") == (Path("/a/b"), 0)
    assert mod.split_run("/a/b@30000") == (Path("/a/b"), 30000)
