# Deployment and Publishing

This guide is for maintainers who verify wheel builds in CI, and for contributors who build the Rust extension locally.

## Contributor develop

Package builds use maturin (Rust + PyO3). Contributors and CI need a Rust toolchain; end users installing a prebuilt wheel do not (tokenizers-style UX).

**Requirements:** [rustup](https://rustup.rs/) stable (`rust-toolchain.toml`) and `maturin` (`pip install -e ".[dev]"` or the build-system pin).

**Canonical contributor workflow:**

```bash
maturin develop --release
pytest
```

## Verify-only wheel CI (current)

CI builds **maturin wheels** on native GitHub runners for **linux, macOS, and Windows** (`PyO3/maturin-action`), uploads wheel artifacts, then installs each OS-matched `.whl` into a **fresh virtualenv** (not editable) and runs the full pytest suite. That path proves what end users install (PACK-01..03).

- **Wheels only** this phase — no sdist artifact requirement (an sdist install would push users toward compiling with a Rust toolchain).
- **One build Python per OS** (3.12) with `abi3-py310`, so one wheel covers Python ≥3.10.
- Architectures = **native runner arches only** (no expanded manylinux/universal2 matrix yet).
- Tag → **PyPI publish remains intentionally disabled** (`release` job `if: false`) until a later deliberate publish decision. CI artifacts are for verification, not a production release gate.

See `.github/workflows/CI.yml`: `test` (develop matrix), `wheels`, `test-wheel`, and hard-skipped `release`.

## Overview (future publish)

When publish is re-enabled, EffDim will use GitHub Actions to upload verified multi-OS wheels to PyPI. Until then, treat the sections below as **planned** maintainer setup — not an active tag→PyPI requirement.

## Prerequisites (when publishing)

### PyPI Account Setup

1. Create a PyPI account at [pypi.org](https://pypi.org/)
2. Generate an API token:
   - Go to Account Settings → API Tokens
   - Click "Add API token"
   - Name: `effdim-github-actions`
   - Scope: `Project: effdim`
3. Copy the token (starts with `pypi-`)

### GitHub Repository Setup

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `PYPI_API_TOKEN`
4. Value: [paste PyPI token]
5. Click **Add secret**

Trusted publishing / attestations can replace a long-lived token when the release job is re-enabled.

## Release Process (disabled until publish decision)

### 1. Update Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.1.1"  # Update this line
```

### 2. Update Changelog

Document changes in `CHANGELOG.md` or release notes.

### 3. Commit Changes

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to 0.1.1"
git push origin main
```

### 4. Create and Push Tag

```bash
git tag -a v0.1.1 -m "Release version 0.1.1"
git push origin v0.1.1
```

Publish only after the `release` job is deliberately re-enabled and wired to the multi-OS wheel artifacts.

### 5. Verify Release

After a successful publish:

```bash
pip install --upgrade effdim
python -c "import effdim; print(effdim.__version__)"
```

## Build Matrix (verify-only today)

Current CI builds **one release wheel per OS** on:

| Platform | Runner | Build Python |
|----------|--------|--------------|
| Linux | `ubuntu-latest` | 3.12 (abi3 ≥3.10) |
| macOS | `macos-latest` | 3.12 (abi3 ≥3.10) |
| Windows | `windows-latest` | 3.12 (abi3 ≥3.10) |

No sdist is produced in the verify-only pipeline. Broader arch matrices (extra manylinux tags, universal2, win_arm64) are deferred.

## Workflow Files

### `.github/workflows/CI.yml`

- **test** — `maturin develop` + pytest on ubuntu/macos × Python 3.10–3.12, plus `cargo test`
- **wheels** — `PyO3/maturin-action@v1` `command: build` with `--release --out dist` on three OS
- **test-wheel** — download artifact → fresh venv → `pip install` wheel + pytest
- **release** — hard-skipped (`if: false`)

### `.github/workflows/publish_docs.yml`

Publishes documentation to GitHub Pages (unchanged).

## Versioning Strategy

EffDim follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (0.x.x → 1.x.x): Breaking API changes
- **MINOR** (x.1.x → x.2.x): New features, backwards compatible
- **PATCH** (x.x.1 → x.x.2): Bug fixes, backwards compatible

## Troubleshooting

### Build Failures

**Rust compilation errors (contributor path):**

```bash
maturin build --release
```

**Clean-env wheel install fails in CI:** Inspect the `wheels` / `test-wheel` jobs and the uploaded `wheels-<os>` artifacts.

### Workflow Not Triggering

Ensure tags use a `v` prefix when publish is eventually re-enabled (`v0.1.1`).

## Security

- Never commit tokens to the repository
- Prefer project-scoped PyPI tokens or trusted publishing when release is enabled
- Dependabot / `cargo audit` / `pip-audit` for dependency scanning

## Additional Resources

- [Maturin Documentation](https://www.maturin.rs/)
- [PyO3/maturin-action](https://github.com/PyO3/maturin-action)
- [PyPI Help](https://pypi.org/help/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)
