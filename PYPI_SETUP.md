# PyPI Publishing Setup

This document explains how to set up automated publishing to PyPI for repository maintainers.

## Prerequisites

1. A PyPI account with permissions to upload to the `effdim` package
2. Repository admin access to configure secrets

## Steps

### 1. Generate PyPI API Token

1. Log in to [PyPI](https://pypi.org/)
2. Go to Account Settings → API Tokens
3. Click "Add API token"
4. Set:
   - Token name: `effdim-github-actions`
   - Scope: `Project: effdim` (or "Entire account" if package doesn't exist yet)
5. Copy the generated token (starts with `pypi-`)

### 2. Add Token to GitHub Secrets

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Set:
   - Name: `PYPI_API_TOKEN`
   - Value: [paste the PyPI token]
4. Click "Add secret"

### 3. Test the Workflow

#### Option 1: Manual Trigger (Recommended for Testing)

1. Go to Actions tab
2. Select "CI" workflow
3. Click "Run workflow"
4. Check the build completes successfully
5. Verify wheels are created (they won't be published without a tag)

#### Option 2: Create a Test Release

1. Update version in `pyproject.toml` (e.g., `0.1.1-beta1`)
2. Commit and push
3. Create a tag:
   ```bash
   git tag v0.1.1-beta1
   git push origin v0.1.1-beta1
   ```
4. Check Actions tab to see the workflow run
5. Verify wheels are built and uploaded to PyPI (test.pypi.org recommended for testing)

## Publishing Process

### For Production Releases

1. Update `version` in `pyproject.toml`
2. Commit the change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```
3. Create and push a version tag:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. The workflow will automatically:
   - Build wheels for all platforms (Linux, macOS, Windows)
   - Build source distribution
   - Upload to PyPI

### What Gets Built

- **Platforms**: 
  - Linux: x86_64, aarch64 (manylinux & musllinux)
  - Windows: x64, x86
  - macOS: x86_64 (Intel, macOS 13), aarch64 (Apple Silicon, macOS 14)
- **Python versions**: 3.8-3.12 (auto-detected via `--find-interpreter`)
- **Total artifacts**: ~40+ wheels + 1 source distribution

## Troubleshooting

### Build Failures

1. Check the Actions tab for error logs
2. Common issues:
   - Rust compilation errors → Fix in `src_rust/lib.rs`
   - Missing dependencies → Update `Cargo.toml`
   - Python compatibility → Check `pyproject.toml`

### Upload Failures

1. Verify `PYPI_API_TOKEN` is correctly set
2. Check token has correct permissions
3. Ensure package name is available on PyPI
4. For first upload, you may need to create the package manually

### Testing Before Release

Use TestPyPI for testing:

1. Get a TestPyPI token from [test.pypi.org](https://test.pypi.org)
2. Add as `TEST_PYPI_API_TOKEN` secret
3. Modify workflow to use TestPyPI URL temporarily
4. Test installation: `pip install --index-url https://test.pypi.org/simple/ effdim`

## Maintenance

- Token expiration: PyPI tokens don't expire, but rotate periodically for security
- Keep the workflow updated with new Python versions as they're released
- Monitor workflow runs for any deprecation warnings
