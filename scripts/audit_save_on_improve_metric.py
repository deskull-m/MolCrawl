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

The output follows the report TSV convention
(docs/07-reports/report_tsv_columns.md): the name begins with ``rt1-`` and the
provenance block is written from the start. The outcome is two columns rather
than one sentence -- ``outcome`` out of a fixed four-word vocabulary, and
``metric_followed`` naming the series when there is one -- so that a reader can
tell what kind of value a cell holds without reading Japanese prose out of it.

Read-only: log_history out of the furthest checkpoint, and the names of the
checkpoint directories. Paths are arguments; this repository is public.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

JUDGED = "eval_loss_mask"
OTHER = "eval_loss"

#: ``outcome`` holds one of these four and nothing else. The Japanese sentences
#: this column used to carry are in the console output and in the report, where a
#: sentence belongs; a column whose values are prose cannot be filtered or counted.
FOLLOWED = "followed"           # one series carries every off-grid checkpoint
NO_OFF_GRID = "no_off_grid_saves"   # the stage wrote nothing off the save_steps grid
SERIES_AGREE = "series_agree"   # the two series pick the same steps: no evidence
INCONCLUSIVE = "inconclusive"   # they differ, but neither covers every checkpoint
OUTCOMES = (FOLLOWED, NO_OFF_GRID, SERIES_AGREE, INCONCLUSIVE)

#: ``eval_points`` counts the evaluation points in the run's history. Both series
#: are logged at the same points, which is checked per run rather than assumed, so
#: one column holds it. It used to be two columns named after the metrics, which
#: said they held a metric's name and held a count instead.
COLUMNS = ["run", "resumed_at", "save_steps", "checkpoints", "off_grid",
           "eval_points", "on_judged", "on_other", "outcome", "metric_followed",
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
    followed = ""
    if not off_grid:
        outcome = NO_OFF_GRID
    elif not discriminates:
        outcome = SERIES_AGREE
    elif len(on_judged) == len(off_grid) and len(on_other) < len(off_grid):
        outcome, followed = FOLLOWED, JUDGED
    elif len(on_other) == len(off_grid) and len(on_judged) < len(off_grid):
        outcome, followed = FOLLOWED, OTHER
    else:
        outcome = INCONCLUSIVE

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
        "eval_points": len(judged),
        # Reported rather than assumed: the single column above is the count of
        # both series only while they are logged at the same points.
        "eval_points_agree": len(judged) == len(other),
        "on_judged": f"{len(on_judged)}/{len(off_grid)}",
        "on_other": f"{len(on_other)}/{len(off_grid)}",
        "outcome": outcome,
        "metric_followed": followed or None,
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
    ap.add_argument("--model", default="several (columns: run)",
                    help="the provenance's model line; the runs given are not all "
                         "one model, so the default names the column that tells "
                         "them apart")
    ap.add_argument("--out", type=Path, required=True,
                    help="the base name begins with rt1- (the column convention)")
    args = ap.parse_args(argv)
    check_name(args.out)

    rows = []
    for spec in args.run:
        run, resumed_at = split_run(spec)
        rows.append(audit(run, args.save_steps, resumed_at))

    # The count column stands for both series, so a run where that is not true is
    # said out loud rather than folded into one number.
    disagreeing = [r["run"] for r in rows if not r["eval_points_agree"]]
    for row in rows:
        del row["eval_points_agree"]
    note = [
        f"eval_points counts the steps at which {JUDGED} was logged; both series "
        f"are logged at the same points, which was true of every run here",
        f"on_judged and on_other are <on that series>/<off_grid>; outcome is one of "
        f"{', '.join(OUTCOMES)} and metric_followed is filled only for {FOLLOWED}",
    ]
    if disagreeing:
        note[0] = (f"eval_points counts the steps at which {JUDGED} was logged; "
                   f"{OTHER} was logged at a different number of points in: "
                   f"{', '.join(disagreeing)}")
    write_tsv(args.out, COLUMNS, rows, {
        "model": args.model,
        "source": "the source column names the trainer_state.json each row was "
                  "read from; log_history and the checkpoint directory names",
        "generated": stamp("scripts/audit_save_on_improve_metric.py"),
        "note": note,
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    if disagreeing:
        print(f"\n** 2 系列の評価点数が違う run: {', '.join(disagreeing)}")

    for row in rows:
        print(f"\n{row['run']}  (step {row['resumed_at']:,} から)")
        print(f"  checkpoint {row['checkpoints']} 本 / うちこの段が書いた "
              f"{args.save_steps} の倍数でないもの {row['off_grid']} 本")
        print(f"  {JUDGED} の改善点に乗る: {row['on_judged']}")
        print(f"  {OTHER} の改善点に乗る:      {row['on_other']}")
        print(f"  判定: {row['outcome']}"
              + (f" ({row['metric_followed']})" if row["metric_followed"] else ""))
        print(f"  best_metric {row['best_metric_recorded']} / "
              f"best_model_checkpoint step {row['best_checkpoint_step']} / "
              f"{JUDGED} の最小 {row['judged_min']} @ {row['judged_min_step']}")
    print(f"\n{len(rows)} run -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
