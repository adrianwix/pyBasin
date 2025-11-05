# Migration Guide

This document explains what changed during the migration to the new UV-based project structure.

## What Changed

### Directory Structure

**Before:**
```
pyBasinWorkspace/
├── pybasin/                    # Library code
├── case_study_duffing_oscillator/
├── case_study_lorenz/
├── case_study_pendulum/
├── case_friction/
└── Pipfile
```

**After:**
```
pyBasinWorkspace/
├── src/pybasin/               # Library code (moved)
├── case_studies/              # Reorganized case studies
│   ├── duffing_oscillator/    # Renamed
│   ├── lorenz/                # Renamed
│   ├── pendulum/              # Renamed
│   ├── friction/              # Renamed
│   └── common/                # Shared utilities
├── tests/                     # New test structure
│   └── integration/
├── docs/                      # Documentation
├── artifacts/                 # Generated outputs
├── scripts/                   # Helper scripts
├── notebooks/                 # Jupyter notebooks
└── pyproject.toml            # Replaces Pipfile
```

### Key Changes

1. **Library Code** (`pybasin/` → `src/pybasin/`)
   - Moved to `src/` layout for better packaging
   - Removed old `setup.py` (replaced by `pyproject.toml`)
   - Package is now properly installable

2. **Case Studies** (renamed and reorganized)
   - `case_study_duffing_oscillator/` → `case_studies/duffing_oscillator/`
   - `case_study_lorenz/` → `case_studies/lorenz/`
   - `case_study_pendulum/` → `case_studies/pendulum/`
   - `case_friction/` → `case_studies/friction/`
   - Added `common/` utilities for shared code

3. **Dependency Management** (`Pipfile` → `pyproject.toml`)
   - Using UV instead of pipenv
   - Proper dependency groups: `[dev]`, `[docs]`, `[case-studies]`
   - Standard pyproject.toml format

4. **Testing Structure**
   - Added `tests/` directory with proper structure
   - Integration tests in `tests/integration/`
   - pytest configuration in `pytest.ini`

5. **Documentation**
   - MkDocs configuration in `mkdocs.yml`
   - Documentation source in `docs/`
   - Can be built with `mkdocs serve`

6. **Artifacts**
   - Centralized `artifacts/` directory for all outputs
   - Subdirectories: `figures/`, `results/`, `reports/`
   - Gitignored by default (regenerable)

## Required Updates

### 1. Import Statements

If you have any absolute imports in your case studies, update them:

**Before:**
```python
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
```

**After:**
```python
from pybasin import BasinStabilityEstimator
# or
from pybasin.BasinStabilityEstimator import BasinStabilityEstimator
```

### 2. File Paths

Update any hardcoded paths to use the new structure:

**Before:**
```python
output_dir = "results"
```

**After:**
```python
from case_studies.common.utils import get_artifacts_dir
output_dir = get_artifacts_dir("results")
```

### 3. Running Code

**Before:**
```bash
pipenv run python case_study_duffing_oscillator/main_supervised.py
```

**After:**
```bash
# Option 1: Use UV
uv run python case_studies/duffing_oscillator/main_supervised.py

# Option 2: Activate venv first
source .venv/bin/activate
python case_studies/duffing_oscillator/main_supervised.py
```

## Migration Checklist

- [x] Created new directory structure
- [x] Moved library code to `src/pybasin/`
- [x] Reorganized case studies
- [x] Created `pyproject.toml` with all dependencies
- [x] Created test structure with integration tests
- [x] Set up documentation with MkDocs
- [x] Updated `.gitignore`
- [x] Installed package with UV
- [x] Created README with new instructions
- [ ] **TODO**: Update imports in case study files
- [ ] **TODO**: Update file paths in case studies
- [ ] **TODO**: Convert case studies to integration tests
- [ ] **TODO**: Add docstrings to library modules
- [ ] **TODO**: Write API documentation

## Next Steps

### Immediate (Required)

1. **Update Imports**: Search for old import patterns and update them
   ```bash
   grep -r "from pybasin\." case_studies/
   ```

2. **Test Case Studies**: Run each case study to ensure they still work
   ```bash
   uv run python case_studies/duffing_oscillator/main_supervised.py
   ```

3. **Fix Any Broken Paths**: Update file I/O to use new artifact structure

### Soon (Recommended)

4. **Write Integration Tests**: Convert case studies to automated tests
   - Implement the test stubs in `tests/integration/`
   - Load MATLAB comparison data
   - Add assertions for basin stability values

5. **Document API**: Add comprehensive docstrings
   - Use Google-style docstrings
   - Document parameters, returns, raises
   - Add usage examples

6. **Build Documentation**: Generate API docs
   ```bash
   uv pip install -e ".[docs]"
   mkdocs serve
   ```

### Later (Nice to Have)

7. **Add More Tests**: Write unit tests for individual components
8. **CI/CD Setup**: Add GitHub Actions for automated testing
9. **Publish Package**: Prepare for PyPI release
10. **Performance Benchmarks**: Compare with MATLAB implementation

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'pybasin'`:
```bash
uv pip install -e .
```

### Case Study Not Running

If case studies fail:
1. Check imports are updated
2. Verify file paths use new structure
3. Ensure virtual environment is activated

### Documentation Build Fails

If MkDocs fails to build:
```bash
uv pip install -e ".[docs]"
```

## Rolling Back

If you need to roll back (not recommended):
```bash
git stash  # Or commit your changes first
git reset --hard HEAD~1  # Go back one commit
```

## Questions?

Check the documentation or open an issue on GitHub.
