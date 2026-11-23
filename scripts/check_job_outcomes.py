"""What happened to the jobs a session submitted, read back from Slurm.

A session that submits long runs and then ends leaves nobody watching them. Four
compounds BERT runs died on a Friday evening and were not noticed until Monday --
56 hours with nothing of that workstream running -- because the only watcher was a
background shell inside the session that submitted them.

Two things are needed and this covers the second: the first is that Slurm itself
records a bad ending (`workflows/on-job-failure.sbatch`), and the second is that a
later session can ask "what did I submit, and how did it end" without having to
remember. Job ids come from a submission record on disk, so the question survives
the session that asked it.

TIMEOUT is not a failure here. A segment cut by the wall clock is the ordinary end of
a chained run and the next segment continues from its checkpoint; the states worth
acting on are FAILED, NODE_FAIL, CANCELLED and OUT_OF_MEMORY.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess

ACTIONABLE = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}
BENIGN = {"COMPLETED", "RUNNING", "PENDING", "COMPLETING", "TIMEOUT"}


def sacct(job_ids):
    """State, name and elapsed for each id, as Slurm reports it."""
    if not job_ids:
        return {}
    out = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
         "-o", "JobID,JobName,State,Elapsed,End"],
        capture_output=True, text=True, timeout=120).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid = parts[0].split(".")[0]
        rows.setdefault(jid, {"job_id": jid, "name": parts[1],
                              "state": parts[2].split()[0] if parts[2] else "",
                              "elapsed": parts[3], "end": parts[4]})
    return rows


def read_record(path):
    """Job ids from a submission record.

    The old form is one `job_id<TAB>label` per line. Newer submitters write a TSV
    with a header, where the id is a `job_id` column and not the first one -- the
    first is when it was submitted. Both are read here, because asking a later
    session to remember which shape a record has defeats the point of the record.
    """
    ids = []
    if not os.path.exists(path):
        return ids
    column = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if not ids and "job_id" in parts and not parts[0].isdigit():
                column = parts.index("job_id")   # a header line
                continue
            if column < len(parts) and parts[column].strip().isdigit():
                ids.append(parts[column].strip())
    return ids


def read_handled(path):
    """{job id: why} from a `job_id<TAB>why` TSV, header optional.

    A job named here is one somebody has already answered for -- resubmitted,
    superseded, or known to be a spare. It is still reported, with the reason, but
    away from the jobs that need an answer, so the count at the top of the report
    means what it says.
    """
    handled = {}
    if not path or not os.path.exists(path):
        return handled
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if not parts[0].strip().isdigit():
                continue          # a header, or a line that names no job
            handled[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else "(理由なし)"
    return handled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="append", default=[],
                    help="a submission record; repeat for several")
    ap.add_argument("--job", action="append", default=[], help="a job id; repeat")
    ap.add_argument("--handled",
                    help="a TSV of job ids already dealt with: `job_id<TAB>why`."
                         " They are reported as handled, with the reason, instead of"
                         " being counted among the ones that need action. Without"
                         " this, a job that was resubmitted days ago keeps heading"
                         " the list every morning, and a real failure arriving later"
                         " joins the bottom of it: the molnl grid lost three days"
                         " that way, with two genuine failures sitting inside a"
                         " standing list of sixty")
    ap.add_argument("--out", help="TSV to write")
    args = ap.parse_args(argv)

    ids = list(args.job)
    for r in args.record:
        ids.extend(read_record(r))
    ids = sorted(set(ids), key=lambda x: (len(x), x))
    if not ids:
        print("  job id が無い。--record か --job を渡す")
        return 2

    rows = sacct(ids)
    missing = [i for i in ids if i not in rows]
    handled = read_handled(args.handled) if args.handled else {}
    actionable = [r for r in rows.values()
                  if r["state"] in ACTIONABLE and r["job_id"] not in handled]
    set_aside = [r for r in rows.values()
                 if r["state"] in ACTIONABLE and r["job_id"] in handled]
    unknown = [r for r in rows.values() if r["state"] not in BENIGN | ACTIONABLE]

    print(f"  照会 {len(ids)} 件 / Slurm が返した {len(rows)} 件")
    for label, group in (("★対処が要る", actionable), ("状態が分からない", unknown)):
        if group:
            print(f"\n  {label}: {len(group)} 件")
            for r in sorted(group, key=lambda x: x["job_id"]):
                print(f"    {r['job_id']:>10s} {r['name'][:38]:38s} {r['state']:14s} "
                      f"{r['elapsed']:>11s} {r['end']}")
    if set_aside:
        print(f"\n  既に対処済み: {len(set_aside)} 件")
        for r in sorted(set_aside, key=lambda x: x["job_id"]):
            print(f"    {r['job_id']:>10s} {r['name'][:38]:38s} {r['state']:14s} "
                  f"{handled[r['job_id']]}")
    if missing:
        print(f"\n  Slurm に記録が無い（古すぎる）: {len(missing)} 件 {missing[:8]}")
    if not actionable and not unknown:
        print("\n  対処が要るものは無い")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["job_id", "name", "state", "elapsed", "end"],
                               delimiter="\t")
            w.writeheader()
            w.writerows(sorted(rows.values(), key=lambda r: r["job_id"]))
        print(f"\n  {args.out}")
    return 1 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(main())
