# Encoder-Scaling Sweep — EleutherAI Pod Runbook

Pod: `universetbd-0`, `ssh root@216.153.49.26`. Everything persistent lives under
`/mnt/ssd-cluster/EffDim` (the pod's `/root` is wiped on every restart).

## 1. Before any SSH

Read `docs/remote-compute/eleutherai-pod-user-guide.md` in full. Then, on the very
first SSH command of the session, verify the remote copy hasn't drifted from the
one you just read:

```bash
ssh root@216.153.49.26 'sha256sum /root/user-guide.md'
sha256sum docs/remote-compute/eleutherai-pod-user-guide.md
```

Compare the two hashes. If they differ, stop, re-fetch the remote guide, and
re-read it before continuing — the rules below assume the guide you read is the
guide in force.

## 2. Setup / after a restart

```bash
ssh root@216.153.49.26
tmux new -As effdim
bash /mnt/ssd-cluster/EffDim/repo/curvature-experiment/sweep/setup_pod.sh
```

**First time only** — the repo is not on the pod yet, so `setup_pod.sh` has
nothing at `$REPO/curvature-experiment/sweep/setup_pod.sh` to run. Get the repo
there first:

```bash
mkdir -p /mnt/ssd-cluster/EffDim
git clone --branch encoder-scaling https://github.com/amanasci/EffDim.git /mnt/ssd-cluster/EffDim/repo
```

This only works if `amanasci/EffDim` is reachable anonymously over HTTPS. Verified
locally on 2026-09-28 with:

```bash
git ls-remote https://github.com/amanasci/EffDim.git
```

which succeeded (listed `refs/heads/main` and other branches) without any
credentials — the repo is public, so the clone above works as-is. `encoder-scaling`
itself will appear in that listing once it's pushed (see §4 of the task-6 report /
step 4 below); if GitHub access from the pod is ever blocked or the repo is made
private later, use the fallback instead:

```bash
# From this machine (akagi's workstation), not the pod:
rsync -a --exclude .venv --exclude 'notebooks/.cache' --exclude 'curvature-experiment/.cache' \
  /home/akagi/Documents/Projects/EffDim/ root@216.153.49.26:/mnt/ssd-cluster/EffDim/repo/
```

If you used the rsync fallback (or the resulting checkout otherwise has no
`origin` remote reachable from the pod), run `setup_pod.sh` with `SKIP_GIT=1` so
it doesn't try to `git fetch`/`merge` a remote it can't see:

```bash
SKIP_GIT=1 bash /mnt/ssd-cluster/EffDim/repo/curvature-experiment/sweep/setup_pod.sh
```

`setup_pod.sh` also auto-detects this case: if `$REPO/.git` exists but `origin`
isn't reachable, it prints a warning and skips the fetch/merge instead of failing,
so `SKIP_GIT=1` is a belt-and-suspenders opt-in, not strictly required.

Record here which path was actually used once you've done it the first time,
so the next restart doesn't need to re-decide:

> _First-time path used: (fill in — "https clone" or "rsync from workstation")._

After the repo is in place (either way), run `setup_pod.sh` as shown at the top
of this section. It is idempotent: re-running it after a restart fast-forwards
the repo (skipped under `SKIP_GIT=1` / no-origin), reuses the existing venv,
reinstalls dependencies (cheap once cached), skips parquets already downloaded,
and re-prints the GPU/CPU diagnostics.

## 3. Choose GPUs and threads

```bash
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv   # busy GPUs (by uuid)
cat /sys/fs/cgroup/cpu.max
```

Use only the GPU indices from `nvidia-smi` that have no processes listed (cross-
reference against the `--query-compute-apps` output, which is keyed by UUID —
match UUIDs back to indices in the main `nvidia-smi` table). Never take a GPU
someone else is using.

Threads: `floor(cgroup CPUs / number of GPUs used)`, minimum 2. Read the CPU
quota from `cpu.max` (`<quota> <period>` in microseconds; CPUs = quota / period)
rather than assuming the pod's nominal 30 cores, since the cgroup may cap it
lower.

## 4. Run

Inside the tmux session from §2:

```bash
cd /mnt/ssd-cluster/EffDim/repo/curvature-experiment
/mnt/ssd-cluster/EffDim/venv/bin/python -m sweep.run_queue \
  --root /mnt/ssd-cluster/EffDim/sweep-out \
  --gpus <list> --threads <n> [--only <substr>] \
  2>&1 | tee -a /mnt/ssd-cluster/EffDim/sweep-out/queue.log
```

Detach with `Ctrl+B, D`. The job keeps running in tmux after you disconnect.

## 5. Progress

```bash
/mnt/ssd-cluster/EffDim/venv/bin/python -m sweep.run_queue \
  --root /mnt/ssd-cluster/EffDim/sweep-out --gpus <list> --threads <n> --dry-run \
  | sort | uniq -c -w4
tail -n 20 /mnt/ssd-cluster/EffDim/sweep-out/logs/<id>.log
```

## 6. Restart recovery

Pods restart without warning and wipe `/root`; `/mnt/ssd-cluster` survives.
After a restart:

1. Re-run §2 (setup) unchanged.
2. Re-run §4 (the same `run_queue` command) unchanged.

`run_queue` skips jobs already marked done and moves aside any partial output
left by a job that was killed mid-write, then re-runs that job from scratch —
no manual cleanup needed.

## 7. Fetch results

From this machine (not the pod):

```bash
rsync -a root@216.153.49.26:/mnt/ssd-cluster/EffDim/sweep-out/{records,arrays,done} \
  curvature-experiment/.cache/scaling/
```

Leave the geometry `.npz` arrays on the pod (`sweep-out/geometry/` or similar) —
they aren't part of this rsync and don't belong in the local record store beyond
what's listed above.

## 8. Rules

Binding for every remote step in this runbook (from
`docs/remote-compute/eleutherai-pod-user-guide.md` and the project's global
constraints):

- Read the local guide in full before the first SSH command of a session; then
  check the remote `/root/user-guide.md` sha256 equals the local file's; if not,
  re-fetch and re-read before continuing.
- Everything persistent under `/mnt/ssd-cluster/EffDim`; nothing we need in `/root`.
- Long jobs only inside `tmux`.
- No concurrent or unbounded `find`/`ls -R` on `/mnt`; direct paths, `-maxdepth`,
  `timeout 30`.
- Never write to `/mnt/datasets` or `/mnt/ssd-1..4`; never touch other users'
  files in `/root`.
- Check `nvidia-smi` and use only GPUs with no other processes; verify the
  effective cgroup CPU limit (`cat /sys/fs/cgroup/cpu.max`) before choosing
  `--threads`.
