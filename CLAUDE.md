# EffDim — working instructions

Research library computing effective dimensionality. `src/effdim/` is the shipped package;
`notebooks/pu_manifold/` is the notebook-scoped helper package for the current milestone,
imported relatively and never from `src/effdim/`. Milestone artifacts live in the gitignored
`notebooks/.cache/`.

## Swiss roll sanity check — required for every new manifold model

**When you implement a new manifold-learning or representation-learning model, you must also
deliver a Swiss roll notebook that tests it.** This applies to any model that maps data to a
lower-dimensional representation and back, or that claims to recover manifold structure —
chart auto-encoders, topological auto-encoders, decoder parameterizations, curvature
estimators, and anything else in that family. It is not optional, and it is not a follow-up
task: the model is not done until its Swiss roll notebook exists and passes.

**Why.** A model that fails on real data has two possible causes: the data has no structure
to find, or the implementation is broken. Only a manifold whose answer is known in advance
separates them. The Swiss roll is that manifold — a 2-dimensional sheet curled into 3-D, the
standard test case, with a correct answer nobody has to argue about. Without this check, a
FAIL on real data is uninterpretable and every downstream conclusion drawn from it is
unsupported.

**Always use the Swiss roll.** Do not substitute another toy dataset. Using the same test
across every model makes the results comparable to each other, which is most of the value.

### What the notebook must do

Name it `notebooks/<phase>_swiss_roll_<model>_check.ipynb`. Reference implementation:
`notebooks/02.2_swiss_roll_cae_check.ipynb` — copy its shape.

1. **Import the model code unchanged** from `notebooks/pu_manifold/`. Never reimplement,
   simplify, or inline a variant of it — the point is to test the code that will actually
   run on real data. If the model only works after you rewrite it for the notebook, that is
   itself the finding.
2. **Generate the data in the notebook** with `sklearn.datasets.make_swiss_roll`
   (~3,000 points, `noise=0.0`, fixed `random_state`). Centre and divide by one global
   standard deviation — a single scalar, so the shape is preserved.
3. **Set the model's latent/chart dimension to 2**, the roll's true intrinsic dimension.
4. **Train from scratch inside the notebook.** Target under two minutes on CPU. Never read
   from or write to `notebooks/.cache/`.
5. **Plot the original and the reconstruction side by side**, both as a 3-D scatter and as an
   x-z scatter — the default 3-D view hides the spiral, the x-z plane shows it
   unambiguously. Colour every plot by the roll's arc-length parameter `t`, so colour bands
   staying in order means the surface stayed in order.
6. **Compare against a matched baseline** — same width, depth, and training protocol, at the
   same 2-D bottleneck. `cae.PlainAutoEncoder` serves for reconstruction models. If the new
   model is not clearly better than the plain baseline on a manifold it was designed for, say
   so plainly; that is a real result about the model.
7. **End with three or four printed pass/fail lines and a one-sentence read-out.** Relative
   error, beats-the-baseline, and whatever structural check the model admits (surviving
   charts, latent rank, persistence agreement).

For a model with no decoder, replace steps 5-6 with the embedding analogue: plot the 2-D
representation and check the spiral unrolls with its colour ordering intact, against a
baseline that is known to succeed.

### What the notebook must not do

No caching, no pre-registration, no git-ancestry proofs, no cfg-hash cache keys, no verdict
JSON artifacts, no threshold tables. Those belong to gated milestone runs on real data. This
is a sanity check: it should be short enough to read in one sitting and cheap enough to re-run
on a whim. If it grows past ~15 cells, it has drifted from its purpose.

## General

- Never delete or rewrite existing notebooks or runner scripts to make room for new work 
  additive only, unless explicitly asked.
- Do not modify `src/effdim/` during the v1.1 milestone.
- Notebooks are committed with their outputs, executed end to end.
- KEEP THINGS SIMPLE FIRST. Only add complexity upon my request. When attempting to solve a problem, focus on the simplest implementation. Then, as issues arise, we will slowly add complexity as needed to solve the problem. Do not try to one shot every edge case and counterexample.

## Remote compute (EleutherAI pod `universetbd-0`, `root@216.153.49.26`)

**Before running ANY command on the SSH server, read `docs/remote-compute/eleutherai-pod-user-guide.md`
in full. First, every session, no exceptions.** It is a verbatim copy of the pod's own
`/root/user-guide.md` (sha256 `5e94dd31…3dc4053b`). If the remote copy has changed, re-fetch it
and re-read before proceeding.

The rules that bite hardest, so they are not missed:

- `/root` is ephemeral and wiped on random, unannounced pod restarts. Clones, venvs, pip
  installs, and outputs go under `/mnt/ssd-cluster` (private, persistent, 500 GB). Nothing we
  care about lives in `/root`.
- Long jobs run inside `tmux`. A foreground job dies with the SSH session.
- No concurrent `find` / `ls -R` on `/mnt/*`; always `-maxdepth` and `timeout`. Heavy I/O can
  hang the NVMe mounts and force a pod restart.
- `/mnt/ssd-1..4` and `/mnt/datasets` are shared with every user on the cluster. Do not write
  to `/mnt/datasets`. Do not delete anything you did not create.
- The pod is a shared root account with other people's projects in `/root`
  (`AstroJEPA`, `Astro-Worldmodels`, `kshitij/`, `translation/`, …). Touch only
  `/mnt/ssd-cluster/EffDim` and paths the phase runbook names.
- Guide says 30 CPU cores by default; `nproc` reports 128. Verify the effective cgroup limit
  before trusting any `--threads` value.

## Spike findings

- **Spike findings for EffDim** (curvature-estimator validation protocol, measured `d=20` dead
  ends, the open saddle-fixture question) → `Skill("spike-findings-effdim")`
