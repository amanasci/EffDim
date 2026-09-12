# EleutherAI GPU Cluster -- New User Guide

You have SSH access to a GPU pod on EleutherAI's Kubernetes cluster, hosted on CoreWeave. This guide covers everything you need to know to get started.

## Connecting

As you probably did recently (since you're reading this document) you can connect to this pod via `ssh root@216.153.49.26`. Access can be authenticated either via a password or via your public key. If you don't have public key authentication set up yet, you can add your public key to `/root/.ssh/authorized_keys` **and** to `/mnt/ssd-cluster/.ssh/authorized_keys`. The former enables passwordless SSH access and the latter is necessary so that if the pod reboots the SSH key will be reloaded automatically.

This pod is named universetbd-0. This is important for you to remember and provide to EleutherAI staff members for help in debugging issues.

## The Single Most Important Thing

**Pods restart randomly and without warning.** CoreWeave pods can be evicted or crash at any time. When this happens, everything in `/root` (your home directory) is wiped. This includes:
- Installed pip/conda packages
- Downloaded repos
- Config files, `.bashrc` customizations
- Anything not on a persistent mount

Unfortunately this means that you will need to checkpoint training runs and otherwise save processes on a regular basis. Depending on your needs it may be a good idea to upload partially trained checkpoints to Hugging Face as you go.

**Put anything you care about on `/mnt/`.** This is non-negotiable. `/mnt/ssd-cluster` is a private 500 GB drive for you to use. Other mounted drives are widely accessible to people in EleutherAI and our collaborators. These drives are large, but please be mindful of the fact that they are shared resources across dozens of people. If you expect to require more than 1 TB of storage, please let us know.

## Computing Resources

### GPUs

Most pods have 8x NVIDIA A100 80GB GPUs. Check what you have with `nvidia-smi`. If you received multiple nodes, the `/job/` directory contains two files: `/job/hostfile` (IPs with slot counts, for DeepSpeed/OpenMPI) and `/job/hosts` (bare IPs, one per line). Every node has SSH access to every other node, but you may not have external SSH access to all of them.

CUDA and PyTorch are pre-installed in the base image. If you plan on doing large scale training, we recommend using our GPT-NeoX library. We have a special build process that sets CoreWeave pods up for training with GPT-NeoX. If it was run for this pod, there should be a `gpt-neox` directory when you initially ssh in. If you didn't receive this and expected to, please let us know. For assistance with large scaling training jobs or the GPT-NeoX library, please reach out to Quentin Anthony on discord or at quentin@eleuther.ai.

### CPU and Memory

Your pod has 30 CPU cores and 500 GB of RAM by default. If you need additional CPU or memory resources, ask your contact.

### Storage Layout

| Path | Persistent? | Shared? | Type | Notes |
|---|---|---|---|---|
| `/root` | NO | No | -- | Ephemeral. Wiped on every pod restart. |
| `/mnt/ssd-cluster` | Yes | No -- your pod only | NVMe | Default 500 GB. Use this for your working data, checkpoints, repos, etc. |
| `/mnt/ssd-1` | Yes | Yes -- all pods | NVMe | General purpose long-term storage for datasets, checkpoints, etc. |
| `/mnt/ssd-2` | Yes | Yes -- all pods | NVMe | Same as above. |
| `/mnt/ssd-3` | Yes | Yes -- all pods | NVMe | Same as above. |
| `/mnt/ssd-4` | Yes | Yes -- all pods | NVMe | Same as above. |
| `/mnt/datasets` | Yes | Yes -- all pods | HDD | Large pretraining datasets. **Do not write to this drive without authorization.** |

Other storage devices exist that are not provided to pods by default. If you believe you should have access to them, please reach out to your contact.

**Rules for shared storage:**

- These are shared across ALL pods and ALL users on the cluster.
- Don't delete other people's data.
- Be mindful of space. Check usage with `df -h /mnt/ssd-1` before writing large datasets.
- Coordinate with others before making large writes (hundreds of GB+) as too many sequential writes can cause the storage device to lock up.
- If there is a large dataset you need, please check that it isn't already on our storage devices. We have the Pile (both raw and shuffled / tokenized exactly as it was seen by Pythia models).

If you need long-term storage or to store an immense amount of data (multiple TB), please speak to Stella about getting that set up.

**Recommendation:** Clone repos and do your day-to-day work on `/mnt/ssd-cluster`. Use `/mnt/ssd-1` etc. for shared datasets and models that multiple people need access to.

### Useful scripts

We provide a script called `upload_checkpoints.py` (located at `/mnt/ssd-1/shared_scripts/upload_checkpoints.py`) that automatically watches for new training checkpoints, uploads them to HuggingFace Hub, and optionally deletes old local copies to free disk space. It is designed to run in the background (inside tmux) alongside your training job.

**Basic usage:**

```bash
# Upload checkpoints as they appear, keeping the 2 most recent locally
python /mnt/ssd-1/shared_scripts/upload_checkpoints.py /mnt/ssd-cluster/my-model MyOrg/my-model

# Keep 5 most recent checkpoints locally instead of 2
python /mnt/ssd-1/shared_scripts/upload_checkpoints.py /mnt/ssd-cluster/my-model MyOrg/my-model --keep 5

# Poll every 10 minutes instead of the default 5
python /mnt/ssd-1/shared_scripts/upload_checkpoints.py /mnt/ssd-cluster/my-model MyOrg/my-model --interval 600

# One-shot mode: upload everything once and exit (no polling)
python /mnt/ssd-1/shared_scripts/upload_checkpoints.py /mnt/ssd-cluster/my-model MyOrg/my-model --once
```

The script expects checkpoint directories named `checkpoint-{step}` (the default for HuggingFace Trainer and many other frameworks). It tracks which checkpoints have already been uploaded via marker files, so it can safely resume after interruptions or pod restarts.

**Requirements:** `pip install huggingface_hub` and either the `HF_TOKEN` environment variable set or a prior `huggingface-cli login`. If your pod was set up by EleutherAI staff, `HF_TOKEN` may already be available in your environment.

**Recommended setup:**

```bash
tmux new -s upload
python /mnt/ssd-1/shared_scripts/upload_checkpoints.py /mnt/ssd-cluster/my-model MyOrg/my-model --keep 3 --log-file /mnt/ssd-cluster/upload.log
# Ctrl+B, D to detach
```

This way the uploader survives SSH disconnects and you can check `upload.log` for progress.

### Installing Software

Pods come with a minimal set of tools pre-installed: git, tmux, htop, vim, nano, curl, wget, rsync, and an SSH server.

You can install anything you want with `pip` or `apt`:

```bash
pip install transformers datasets accelerate
apt update && apt install -y <package>
```

**Remember: these installations live in `/root` and will be lost on pod restart.** For packages you need repeatedly, consider one of these strategies:

1. **Put a setup script on persistent storage:**
   ```bash
   # Save this to /mnt/ssd-cluster/setup.sh
   pip install transformers datasets accelerate wandb
   ```
   Run it after each restart.

2. **Use a conda/venv on persistent storage:**
   ```bash
   # One-time setup
   conda create -p /mnt/ssd-cluster/myenv python=3.10

   # After each restart
   conda activate /mnt/ssd-cluster/myenv
   ```

3. **Install into a persistent location with pip:**
   ```bash
   pip install --target=/mnt/ssd-cluster/pip_packages transformers
   export PYTHONPATH=/mnt/ssd-cluster/pip_packages:$PYTHONPATH
   ```

## Long-Running Jobs: Use tmux

Since your SSH connection will drop if your laptop sleeps or your network hiccups, always run long jobs inside `tmux`:

```bash
# Start a new session
tmux new -s training

# Detach: press Ctrl+B, then D

# Reattach after reconnecting
tmux attach -t training

# List sessions
tmux ls
```
This is especially important because if your SSH connection dies while a job is running in the foreground (without tmux), the job dies too.

Quick tmux reference:
- `Ctrl+B, D` -- detach from session
- `Ctrl+B, C` -- new window
- `Ctrl+B, N` -- next window
- `Ctrl+B, P` -- previous window
- `Ctrl+B, [` -- scroll mode (q to exit)

`screen` is an alternative if you prefer it (`apt install screen` -- but remember, this won't survive a pod restart, so `tmux` which is pre-installed is the better choice).

## Filesystem Safety

**Do not run multiple concurrent `find` or recursive `ls` commands on `/mnt/` filesystems.** The NVMe mounts can become unresponsive under heavy I/O, causing processes to enter an unkillable state that requires a full pod restart. Specifically:

- Run one `find` or `ls -R` at a time on mounted filesystems.
- Use `--maxdepth` and narrow paths: `find /mnt/ssd-1/mydata -maxdepth 2 -name '*.pt'`
- Set timeouts: `timeout 30 find /mnt/ssd-1 -maxdepth 2 -name '*.py'`
- If a command hangs, do NOT launch more filesystem searches. Wait or ask for help.

## Handling Pod Restarts

When your pod restarts (you will notice because your SSH connection drops and you lose everything in `/root`):
1. SSH back in with the same IP address. (The IP usually stays the same, but if it doesn't work, ask your contact to check.)
2. Re-run any setup scripts you need (pip installs, conda activation, etc.)
3. Your data on `/mnt/` will still be there.
4. Check `/etc/motd` when you log in -- it shows your pod's boot history so you can see how many times it has restarted.

## Quick Reference

```bash
# Check GPUs
nvidia-smi

# Check disk space
df -h /mnt/ssd-cluster /mnt/ssd-1 /mnt/ssd-2

# Check what's running
htop

# Start a tmux session
tmux new -s work

# Reattach to tmux
tmux attach -t work
```

## Getting Help

If something is broken (SSH not working, pod not starting, GPUs not visible), reach out to your contact. You can also ask in **#behind-the-scenes** on the EleutherAI Discord (if you aren't in that channel, ask your contact to add you). When reporting an issue, include:
- Your pod name (check with `hostname`)
- What you were doing
- Any error messages