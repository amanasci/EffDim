# Deployment and Publishing

This guide is for maintainers who publish releases to PyPI.

## Overview

EffDim uses GitHub Actions to automatically build and publish prebuilt wheels for multiple platforms and Python versions.

## Prerequisites

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

## Release Process

### 1. Update Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.1.1"  # Update this line
```

### 2. Update Changelog

Document changes in `CHANGELOG.md` or release notes:

```markdown
## [0.1.1] - 2024-01-23

### Added
- New feature X
- Performance improvements

### Fixed
- Bug in function Y
```

### 3. Commit Changes

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to 0.1.1"
git push origin main
```

### 4. Create and Push Tag

```bash
# Create annotated tag
git tag -a v0.1.1 -m "Release version 0.1.1"

# Push tag to trigger workflow
git push origin v0.1.1
```

### 5. Monitor Build

1. Go to **Actions** tab in GitHub
2. Watch the "Build and Publish to PyPI" workflow
3. Verify all jobs complete successfully:
   - ✅ Build wheels (Linux, macOS, Windows)
   - ✅ Build source distribution
   - ✅ Publish to PyPI

### 6. Verify Release

After successful workflow:

```bash
# Wait a few minutes for PyPI to update
pip install --upgrade effdim

# Verify version
python -c "import effdim; print(effdim.__version__)"
```

## Build Matrix

The CI workflow builds wheels for:

### Platforms and Architectures

#### Linux
- **manylinux**: x86_64, aarch64
- **musllinux**: x86_64, aarch64

#### Windows
- x64 (64-bit)
- x86 (32-bit)

#### macOS
- x86_64 (Intel) - macOS 13+
- aarch64 (Apple Silicon) - macOS 14+

### Python Versions

The workflow uses `--find-interpreter` to automatically build for all available Python versions (3.8-3.12) on each platform.

### Total Artifacts

Approximately **40+ wheels** + 1 source distribution per release, covering all combinations of platforms, architectures, and Python versions.

## Workflow Files

### `.github/workflows/CI.yml`

The main CI/CD workflow based on maturin's recommended structure:

- Triggered by: Pushes to main/master, PRs, tags, or manual dispatch
- Separate jobs for: linux, musllinux, windows, macos, sdist, release
- Publishes: To PyPI on version tags (automatically)

Key features:

- Uses [PyO3/maturin-action](https://github.com/PyO3/maturin-action)
- Enables sccache for faster builds (disabled on release tags)
- Builds manylinux and musllinux for maximum compatibility
- Unique artifact naming prevents conflicts
- Includes build attestations for security

### `.github/workflows/publish_docs.yml`

Publishes documentation to GitHub Pages (unchanged).

## Testing Before Release

### Option 1: Manual Workflow Trigger

1. Go to **Actions** → **Build and Publish to PyPI**
2. Click **Run workflow**
3. Select branch
4. Review artifacts (won't publish without tag)

### Option 2: Test with TestPyPI

Modify workflow temporarily to use TestPyPI:

```yaml
- name: Publish to TestPyPI
  uses: PyO3/maturin-action@v1
  env:
    MATURIN_PYPI_TOKEN: ${{ secrets.TEST_PYPI_API_TOKEN }}
  with:
    command: upload
    args: --non-interactive --repository-url https://test.pypi.org/legacy/ dist/*
```

Then install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple effdim
```

## Versioning Strategy

EffDim follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (0.x.x → 1.x.x): Breaking API changes
- **MINOR** (x.1.x → x.2.x): New features, backwards compatible
- **PATCH** (x.x.1 → x.x.2): Bug fixes, backwards compatible

### Pre-releases

For beta/RC versions:

```toml
version = "0.2.0b1"  # Beta 1
version = "0.2.0rc1" # Release Candidate 1
```

Tag as: `v0.2.0b1`, `v0.2.0rc1`

## Troubleshooting

### Build Failures

**Rust compilation errors:**

```bash
# Check locally
maturin build --release

# View logs in GitHub Actions
```

**Python compatibility issues:**

- Ensure `requires-python` in `pyproject.toml` matches tested versions
- Check minimum Rust version in `Cargo.toml`

### Upload Failures

**Invalid token:**

- Regenerate PyPI token
- Update `PYPI_API_TOKEN` secret in GitHub

**Package name conflict:**

- First release must be manually created on PyPI
- Or use different package name

**File already exists:**

- Can't re-upload same version
- Bump version and retry
- Use `--skip-existing` flag (already in workflow)

### Workflow Not Triggering

**Tag format issues:**

```bash
# Correct
git tag v0.1.1

# Incorrect
git tag 0.1.1  # Missing 'v' prefix
```

**Branch protection:**

- Ensure tags can be pushed to repository
- Check branch protection rules

## Rollback Procedure

If a bad release is published:

### Option 1: Yank Release (Recommended)

On PyPI:

1. Go to project page
2. Click on problematic version
3. Click "Options" → "Yank release"
4. Publish fixed version

### Option 2: Delete Release

⚠️ **Not recommended** - breaks existing installations

```bash
# Delete tag locally
git tag -d v0.1.1

# Delete tag remotely
git push origin :refs/tags/v0.1.1
```

Then publish corrected version.

## Security

### API Token Management

- **Rotate tokens** periodically (every 6-12 months)
- **Use project-scoped tokens** (not account-wide)
- **Never commit tokens** to repository
- **Store in GitHub Secrets** only

### Dependency Security

Automated security scanning:

- Dependabot alerts (GitHub)
- `cargo audit` for Rust dependencies
- `pip-audit` for Python dependencies

## Maintenance Checklist

Before each release:

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Dependencies updated
- [ ] Security audit clean
- [ ] Performance benchmarks run
- [ ] Breaking changes documented

## Additional Resources

- [Maturin Documentation](https://www.maturin.rs/)
- [PyPI Help](https://pypi.org/help/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)
