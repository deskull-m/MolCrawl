#!/bin/bash
#SBATCH --job-name=mc-soi-metric-audit
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read back which metric a run's off-grid checkpoints followed.
#
# No GPU: log_history out of the furthest checkpoint, and the checkpoint directory
# names. It runs here rather than on the login node because compute there is not
# allowed, and a 45,000-step history is read per run.
#
#   sbatch --export=ALL,SOI_RUNS='<dir> <dir> ...',SOI_OUT=<file.tsv>[,SOI_SAVE_STEPS=1000] \
#          workflows/save-on-improve-metric-audit.sbatch
#
# The run list and the output are arguments with no default: the run directories and
# the report directory are outside every checkout, and this repository is public.
set -uo pipefail

: "${SOI_RUNS:?set SOI_RUNS to a space-separated list of run directories}"
: "${SOI_OUT:?set SOI_OUT to the TSV to write}"
SAVE_STEPS="${SOI_SAVE_STEPS:-1000}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# A worktree holds only tracked files, so the interpreter is not inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

# Each entry is DIR or DIR@RESUME_STEP; the directory is checked without the tag.
ARGS=()
for spec in ${SOI_RUNS}; do
  dir="${spec%@*}"
  [ -d "${dir}" ] || { echo "no such run directory: ${dir}" >&2; exit 1; }
  ARGS+=(--run "${spec}")
done

echo "=== save-on-improve metric audit on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:       ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "save_steps: ${SAVE_STEPS}"
echo "output:     ${SOI_OUT}"
"${PYTHON}" scripts/audit_save_on_improve_metric.py \
    "${ARGS[@]}" --save-steps "${SAVE_STEPS}" --out "${SOI_OUT}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
