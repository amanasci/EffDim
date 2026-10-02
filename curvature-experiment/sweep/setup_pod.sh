#!/usr/bin/env bash
# Idempotent environment setup for the encoder-scaling sweep on the EleutherAI pod.
# Everything lives under /mnt/ssd-cluster/EffDim (the pod's /root is wiped on restart) --
# including uv itself and the Python interpreter it installs: the pod's system python3
# is 3.10, which doesn't match our pins (the project venv this repo was developed against
# is 3.14 -- verify with `.venv/bin/python --version` on your workstation and update
# PYTHON_VERSION below if that ever changes), so we let uv fetch and manage the right
# interpreter on persistent storage instead of relying on the pod's system python3.
set -euo pipefail
BASE=/mnt/ssd-cluster/EffDim
REPO=$BASE/repo; VENV=$BASE/venv; OUT=$BASE/sweep-out; HF=$BASE/hf-cache
BRANCH=${BRANCH:-encoder-scaling}
QM9_OUT=$BASE/qm9-out
MANIFEST=curvature-experiment/encoders.yaml
while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST=$2; shift 2 ;;
    *) echo "error: unknown argument $1 (usage: setup_pod.sh [--manifest <path>])" >&2; exit 2 ;;
  esac
done
case "$MANIFEST" in /*) ;; *) MANIFEST=$REPO/$MANIFEST ;; esac
SNAPSHOT=bc081f8a5db4767edcd958653d96efde9137de0b
# Must match the local venv's Python (`.venv/bin/python --version`), not the pod's system
# python3 (3.10). Verified locally on 2026-09-28.
PYTHON_VERSION=${PYTHON_VERSION:-3.14}
# PyTorch CUDA wheel index for torch 2.13.0 on linux x86_64. Verified 2026-09-28 against
# https://download.pytorch.org/whl/<tag>/torch/: cu121 tops out at 2.5.1, cu124 tops out
# at 2.6.0 (neither serves 2.13.0); cu126 and cu129 both serve 2.13.0. cu126 is used here
# for the broadest driver compatibility with the pod's A100s; override with TORCH_CUDA_TAG
# if the pod's driver requires otherwise.
TORCH_CUDA_TAG=${TORCH_CUDA_TAG:-cu126}
mkdir -p "$BASE" "$OUT" "$HF" "$QM9_OUT"

echo "--- disk space on /mnt/ssd-cluster (before anything else) ---"
df -h /mnt/ssd-cluster
AVAIL_GB=$(df --output=avail -BG /mnt/ssd-cluster | tail -n1 | tr -dc '0-9')
# The 30 GB is for the first download of the 31 parquets (~21 GB) plus the venv. On a
# rerun with every parquet already on disk that space is spent, so only headroom for the
# dependency reinstall is needed.
N_PARQUETS=$( (ls "$OUT"/hf/physics/*_test.parquet 2>/dev/null || true) | wc -l)
NEED_GB=30
if [ "$N_PARQUETS" -ge 31 ]; then NEED_GB=5; fi
if [ "${AVAIL_GB:-0}" -lt "$NEED_GB" ]; then
  echo "error: only ${AVAIL_GB:-0} GB free on /mnt/ssd-cluster; need at least $NEED_GB GB before" >&2
  echo "downloading parquets and installing dependencies ($N_PARQUETS of 31 parquets present)." >&2
  echo "Free up space (or ask about a bigger allocation) and re-run." >&2
  exit 1
fi

# uv itself, its managed Python interpreters, and its download/wheel cache all live under
# $BASE -- nothing installed to /root, which is wiped on restart. Besides the binary
# itself, the stock installer also (unless told otherwise) writes an install receipt to
# ${XDG_CONFIG_HOME:-$HOME/.config}/uv and, without UV_NO_MODIFY_PATH, edits shell rc
# files under $HOME -- both of which would land in /root. UV_NO_MODIFY_PATH=1 already
# suppresses the rc-file edits; UV_UNMANAGED_INSTALL (checked in the installer source,
# astral.sh/uv/install.sh, on 2026-09-28) additionally disables the self-updater/receipt
# entirely (it forces the same install dir we already set via UV_INSTALL_DIR, since that
# variable takes precedence, so it doesn't change where uv itself lands). XDG_CONFIG_HOME
# is exported too as a second, independent line of defense in case any future installer
# version writes config there regardless.
export UV_INSTALL_DIR=$BASE/uv
export UV_PYTHON_INSTALL_DIR=$BASE/uv-python
export UV_CACHE_DIR=$BASE/uv-cache
export XDG_CONFIG_HOME=$BASE/xdg-config
mkdir -p "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$XDG_CONFIG_HOME"
UV=$UV_INSTALL_DIR/uv
if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" \
    UV_UNMANAGED_INSTALL="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 XDG_CONFIG_HOME="$XDG_CONFIG_HOME" sh
fi

# Git: SKIP_GIT=1 (or a repo checkout with no reachable "origin") skips fetch/merge
# entirely, for the case where the repo was rsync'd onto the pod instead of cloned
# (e.g. the GitHub repo is private and the pod has no credentials for it).
if [ "${SKIP_GIT:-0}" = "1" ]; then
  echo "SKIP_GIT=1: leaving \$REPO's git state untouched (rsync-deployed checkout)."
elif [ -d "$REPO/.git" ]; then
  if git -C "$REPO" remote get-url origin >/dev/null 2>&1 && git -C "$REPO" fetch -q origin "$BRANCH"; then
    git -C "$REPO" checkout -q "$BRANCH"
    git -C "$REPO" merge -q --ff-only "origin/$BRANCH"
  else
    echo "warning: $REPO/.git has no reachable 'origin' for $BRANCH; skipping fetch/merge." >&2
    echo "warning: re-run with SKIP_GIT=1 to silence this once you know that's expected." >&2
  fi
else
  if ! git clone -q --branch "$BRANCH" https://github.com/amanasci/EffDim.git "$REPO"; then
    echo "error: git clone failed and $REPO does not exist yet." >&2
    echo "If the pod cannot reach/authenticate to GitHub, rsync the repo from your workstation instead:" >&2
    echo "  rsync -a --exclude .venv --exclude 'notebooks/.cache' --exclude 'curvature-experiment/.cache' \\" >&2
    echo "    /home/akagi/Documents/Projects/EffDim/ root@216.153.49.26:$REPO/" >&2
    echo "then re-run this script with SKIP_GIT=1." >&2
    exit 1
  fi
fi

"$UV" python install "$PYTHON_VERSION"
if [ ! -x "$VENV/bin/python" ]; then "$UV" venv --python "$PYTHON_VERSION" "$VENV"; fi
TORCH_PIN=$(grep -E '^torch==' "$REPO/curvature-experiment/requirements.txt" | sed 's/+cpu//')
# Drop the torch== pin (installed separately from the CUDA index below) AND every
# extra/alternate package-index line (--extra-index-url, --index-url, -i, --find-links,
# -f). requirements.txt carries `--extra-index-url .../whl/cpu` so pip can resolve
# torch's `+cpu` local version; left in here, `uv pip install` would also be free to take
# OTHER packages from that CPU wheel index (uv resolves each package from the first index
# that has it, not necessarily the newest match), which is how `requests` was previously
# pinned down to the stale 2.28.1 build on that index and broke datasets>=5.0.1's
# requests>=2.32.2 requirement. requirements.txt always has non-index, non-torch lines
# (the actual dependency pins), so grep -v never matches everything; `|| true` is kept
# only as a defensive guard against that under `set -euo pipefail`.
grep -vE '^torch==|^--extra-index-url|^--index-url|^-i |^--find-links|^-f ' \
  "$REPO/curvature-experiment/requirements.txt" > "$BASE/requirements-gpu.txt" || true
"$UV" pip install --python "$VENV/bin/python" -q -r "$BASE/requirements-gpu.txt"
"$UV" pip install --python "$VENV/bin/python" -q "$TORCH_PIN" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_TAG"
# effdim itself (sweep/intrinsic_dim.py imports effdim.geometry); --no-deps so it cannot move a pinned dependency
"$UV" pip install --python "$VENV/bin/python" -q --no-deps -e "$REPO"

# HF_HOME/HF_HUB_CACHE stay on /mnt so any metadata huggingface_hub keeps outside the
# hf_hub_download(..., local_dir=...) calls below also lands on persistent storage, not
# /root. The parquet downloads themselves do NOT double this data into that cache: as of
# huggingface_hub >= 0.23 (this pod's pinned version, verified locally on 2026-09-28),
# passing `local_dir` bypasses the blob/symlink cache entirely -- the file is written
# straight into local_dir, and only a small `.cache/huggingface/` metadata dir (not a
# second copy of the parquet) is created at the root of local_dir itself. So there is
# nothing to de-duplicate here; if a future huggingface_hub version changes that
# behaviour, re-check this comment and, if needed, delete the cached blob after copying
# or move to `local_dir` + `HF_HUB_CACHE` pointed at a throwaway dir you clean up.
export HF_HOME=$HF
export HF_HUB_CACHE=$HF/hub
"$VENV/bin/python" - "$REPO" "$OUT" "$SNAPSHOT" "$MANIFEST" "$QM9_OUT" <<'EOF'
import hashlib
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, out, snap, manifest, qm9_out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
sys.path.insert(0, str(repo / "curvature-experiment"))
from sweep.manifest import load_manifest
m = load_manifest(manifest)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if m.labels is not None:
    # molecules: no galaxy snapshot or parquet download; the table and any embeddings are checked by sha256
    if not Path(m.label_table).exists():
        raise SystemExit(f"molecule table missing on the pod: {m.label_table}")
    got = sha256(m.label_table)
    if got != m.label_table_sha256:
        raise SystemExit(f"molecule table sha256 {got} != manifest {m.label_table_sha256}")
    print(f"molecule table sha256 verified: {got}")
    ok, unpinned, missing = [], [], []
    for e in m.encoders:
        p = qm9_out / "hf" / e.parquet_file
        if not p.exists():
            missing.append(e.name)
        elif e.parquet_sha256 is None:
            unpinned.append(e.name)
        elif sha256(p) != e.parquet_sha256:
            raise SystemExit(f"embedding sha256 mismatch for {e.name}: {p}")
        else:
            ok.append(e.name)
    print(f"embeddings verified {len(ok)} of {len(m.encoders)}; present but not yet pinned: {', '.join(unpinned) or 'none'}; "
          f"missing: {', '.join(missing) or 'none'}")
else:
    assert m.snapshot == snap, (m.snapshot, snap)
    for e in m.encoders:
        dest = out / "hf" / e.parquet_file
        if dest.exists() and dest.stat().st_size > 0:
            continue
        hf_hub_download(m.repo, e.parquet_file, repo_type="dataset", revision=snap, local_dir=str(out / "hf"))
        print("downloaded", e.parquet_file, flush=True)
    if not Path(m.label_table).exists():
        raise SystemExit(f"label table missing on the pod: {m.label_table}")
    print("parquets and label table present")
EOF
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# cgroup v1 pods (this cluster) have no /sys/fs/cgroup/cpu.max; fall back to the v1
# quota/period files. quota=-1 means unlimited, in which case the pod-guide's nominal
# 30-core default is the budget to divide across GPUs (see POD_RUNBOOK.md section 3).
# Each `cat` below is individually guarded (`2>/dev/null || echo ...`) rather than
# relying on the outer `||` alone: under `set -e`, a failing command substitution (e.g.
# cpu.cfs_quota_us also missing, on some other cgroup setup) would otherwise abort the
# whole script even though this is purely diagnostic output.
cat /sys/fs/cgroup/cpu.max 2>/dev/null || {
  q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo "?")
  p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo "?")
  echo "quota=$q period=$p"
}
