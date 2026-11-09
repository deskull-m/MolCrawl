"""Which metric did a run's off-grid checkpoints actually follow?

``save_on_improve`` asks for a checkpoint whenever the judged metric reaches a new
best, so a run writes checkpoints on two schedules: the periodic ``save_steps``
grid, and the improvement points. The improvement points are the evidence, because
only they depend on a metric: a checkpoint off the grid exists because something
improved, and *which* series it was can be read back by asking which series has a
new best at that step.

This matters because the callback has tracked the wrong series before. Before
``c792b7e7`` (2026-10-16) ``fallback_metric`` defaulted to ``eval_loss``, and
``eval_loss_mask`` is not in the metrics at ``on_evaluate`` -- ``MlmBreakdownMixin``
adds it through ``self.log()`` after ``super().evaluate()`` returns -- so runs from
before that commit saved on the overall loss while printing the mask metric's name.

Two series are therefore scored for each run and the answer is whichever one the
off-grid checkpoints sit on. The run is also reported as *undecidable* when the two
series pick the same steps: agreement is not evidence then, and saying so is the
point of the column.

The best is taken as restarting where this stage's own training began, which is how
the callback behaves: it is constructed fresh when training starts, so a
continuation has no memory of the previous stage's minimum.

**Where that is has to be said, not guessed.** HF restores ``log_history`` from the
checkpoint a run resumed from, and appends to it; the restored rows and the run's
own rows are one ascending series with no marker between them. Taking the first
step in the history would therefore start the best at the previous stage's
minimum, and none of this stage's saves would look like an improvement. A run that
continued another is written ``DIR@STEP``, where STEP is the step it resumed from.

    audit_save_on_improve_metric.py --run DIR[@RESUME_STEP] [--run ...] --out FILE

Read-only: log_history out of the furthest checkpoint, and the names of the
checkpoint directories. Paths are arguments; this repository is public.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

JUDGED = "eval_loss_mask"
OTHER = "eval_loss"

COLUMNS = ["run", "resumed_at", "save_steps", "checkpoints", "off_grid",
           "judged_metric", "other_metric", "on_judged", "on_other", "verdict",
           "best_metric_recorded", "best_checkpoint_step", "judged_min_step",
           "judged_min", "source"]


def split_run(spec: str) -> tuple[Path, int]:
    """``DIR`` or ``DIR@RESUME_STEP``.

    The resume step cannot be read out of the run, so it is given. Without it a
    continuation's best would start at the previous stage's minimum.
    """
    if "@" in spec:
        head, _, tail = spec.rpartition("@")
        return Path(head), int(tail)
    return Path(spec), 0


def newest_trainer_state(run: Path) -> Path:
    steps = sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.is_dir())
    if not steps:
        raise SystemExit(f"no checkpoint-* directory under {run}")
    path = run / f"checkpoint-{steps[-1]}" / "trainer_state.json"
    if not path.exists():
        raise SystemExit(f"no trainer_state.json in {path.parent}")
    return path


def series(log_history, metric) -> dict[int, float]:
    """{step: value} for one metric. A step logged twice keeps the last value."""
    out = {}
    for entry in log_history:
        if metric in entry and entry.get("step") is not None:
            out[int(entry["step"])] = float(entry[metric])
    return out


def improvement_steps(points: dict[int, float], start: int = 0) -> set[int]:
    """The steps at which the series reaches a new low, counting from `start`."""
    best, out = None, set()
    for step in sorted(points):
        if step < start:
            continue
        if best is None or points[step] < best:
            best = points[step]
            out.add(step)
    return out


def checkpoint_steps(run: Path) -> list[int]:
    return sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.is_dir())


def audit(run: Path, save_steps: int, resumed_at: int = 0) -> dict:
    state_path = newest_trainer_state(run)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])

    judged = series(history, JUDGED)
    other = series(history, OTHER)
    # Strictly after the resume step: the checkpoint's own step belongs to the
    # stage before this one.
    own = [s for s in judged if s > resumed_at]
    first_eval = min(own) if own else (min(judged) if judged else 0)

    kept = checkpoint_steps(run)
    off_grid = [s for s in kept if s % save_steps and s > resumed_at]

    j_steps = improvement_steps(judged, first_eval)
    o_steps = improvement_steps(other, first_eval)
    on_judged = [s for s in off_grid if s in j_steps]
    on_other = [s for s in off_grid if s in o_steps]

    # The two series must disagree somewhere for the comparison to mean anything.
    discriminates = j_steps != o_steps
    if not off_grid:
        verdict = "追加保存なし"
    elif not discriminates:
        verdict = "判定不能（2 系列が同じ step を選ぶ）"
    elif len(on_judged) == len(off_grid) and len(on_other) < len(off_grid):
        verdict = JUDGED
    elif len(on_other) == len(off_grid) and len(on_judged) < len(off_grid):
        verdict = OTHER
    else:
        verdict = "どちらとも言えない"

    best_ck = state.get("best_model_checkpoint") or ""
    # The field holds an absolute path written at train time; only its step is read,
    # because the tree it names may no longer exist.
    best_step = best_ck.rsplit("-", 1)[1] if "-" in best_ck else ""
    min_step = min(judged, key=lambda s: judged[s]) if judged else ""

    return {
        "run": run.name,
        "resumed_at": resumed_at,
        "save_steps": save_steps,
        "checkpoints": len(kept),
        "off_grid": len(off_grid),
        "judged_metric": len(judged),
        "other_metric": len(other),
        "on_judged": f"{len(on_judged)}/{len(off_grid)}",
        "on_other": f"{len(on_other)}/{len(off_grid)}",
        "verdict": verdict,
        "best_metric_recorded": ("" if state.get("best_metric") is None
                                 else f"{state['best_metric']:.6f}"),
        "best_checkpoint_step": best_step,
        "judged_min_step": min_step,
        "judged_min": f"{judged[min_step]:.6f}" if judged else "",
        "source": f"{run.name}/{state_path.parent.name}/{state_path.name}",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, metavar="DIR[@STEP]",
                    help="a run directory, with @STEP if it continued another; "
                         "repeat for several")
    ap.add_argument("--save-steps", type=int, default=1000,
                    help="the periodic save interval the run was configured with")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for spec in args.run:
        run, resumed_at = split_run(spec)
        rows.append(audit(run, args.save_steps, resumed_at))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"\n{row['run']}  (step {row['resumed_at']:,} から)")
        print(f"  checkpoint {row['checkpoints']} 本 / うちこの段が書いた "
              f"{args.save_steps} の倍数でないもの {row['off_grid']} 本")
        print(f"  {JUDGED} の改善点に乗る: {row['on_judged']}")
        print(f"  {OTHER} の改善点に乗る:      {row['on_other']}")
        print(f"  判定: {row['verdict']}")
        print(f"  best_metric {row['best_metric_recorded']} / "
              f"best_model_checkpoint step {row['best_checkpoint_step']} / "
              f"{JUDGED} の最小 {row['judged_min']} @ {row['judged_min_step']}")
    print(f"\n{len(rows)} run -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
