#!/usr/bin/env bash
# Idempotent environment setup for the encoder-scaling sweep on the EleutherAI pod.
# Everything lives under /mnt/ssd-cluster/EffDim (the pod's /root is wiped on restart).
set -euo pipefail
BASE=/mnt/ssd-cluster/EffDim
REPO=$BASE/repo; VENV=$BASE/venv; OUT=$BASE/sweep-out; HF=$BASE/hf-cache
BRANCH=${BRANCH:-encoder-scaling}
SNAPSHOT=bc081f8a5db4767edcd958653d96efde9137de0b
# PyTorch CUDA wheel index for torch 2.13.0 on linux x86_64. Verified 2026-09-28 against
# https://download.pytorch.org/whl/<tag>/torch/: cu121 tops out at 2.5.1, cu124 tops out
# at 2.6.0 (neither serves 2.13.0); cu126 and cu129 both serve 2.13.0. cu126 is used here
# for the broadest driver compatibility with the pod's A100s; override with TORCH_CUDA_TAG
# if the pod's driver requires otherwise.
TORCH_CUDA_TAG=${TORCH_CUDA_TAG:-cu126}
mkdir -p "$BASE" "$OUT" "$HF"

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

if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV"; fi
TORCH_PIN=$(grep -E '^torch==' "$REPO/curvature-experiment/requirements.txt" | sed 's/+cpu//')
grep -vE '^torch==' "$REPO/curvature-experiment/requirements.txt" > "$BASE/requirements-gpu.txt"
"$VENV/bin/pip" install -q -r "$BASE/requirements-gpu.txt"
"$VENV/bin/pip" install -q "$TORCH_PIN" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_TAG"
export HF_HOME=$HF
"$VENV/bin/python" - "$REPO" "$OUT" "$SNAPSHOT" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, out, snap = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(repo / "curvature-experiment"))
from sweep.manifest import load_manifest
m = load_manifest()
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
cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "cpu.max not readable"
