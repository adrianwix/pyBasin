#!/usr/bin/env python3
"""
Migration Summary Script

This script helps verify the migration to the new project structure.
Run this after migration to ensure everything is in place.
"""

from pathlib import Path
import sys


def check_directory_structure():
    """Check if all expected directories exist."""
    project_root = Path(__file__).parent.parent
    
    expected_dirs = [
        "src/pybasin",
        "case_studies/duffing_oscillator",
        "case_studies/lorenz",
        "case_studies/pendulum",
        "case_studies/friction",
        "case_studies/common",
        "tests",
        "tests/integration",
        "docs",
        "docs/getting-started",
        "docs/case-studies",
        "artifacts",
        "artifacts/figures",
        "artifacts/results",
        "artifacts/reports",
        "scripts",
        "notebooks/examples",
    ]
    
    print("📁 Checking directory structure...")
    missing = []
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} (MISSING)")
            missing.append(dir_path)
    
    return len(missing) == 0


def check_key_files():
    """Check if key files exist."""
    project_root = Path(__file__).parent.parent
    
    key_files = [
        "pyproject.toml",
        "README.md",
        "mkdocs.yml",
        "pytest.ini",
        ".gitignore",
        "src/pybasin/__init__.py",
        "tests/conftest.py",
        "docs/index.md",
    ]
    
    print("\n📄 Checking key files...")
    missing = []
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (MISSING)")
            missing.append(file_path)
    
    return len(missing) == 0


def check_library_modules():
    """Check if library modules exist in src/pybasin."""
    project_root = Path(__file__).parent.parent
    src_pybasin = project_root / "src" / "pybasin"
    
    expected_modules = [
        "ASBasinStabilityEstimator.py",
        "ASPlotter.py",
        "BasinStabilityEstimator.py",
        "ClusterClassifier.py",
        "FeatureExtractor.py",
        "ODESystem.py",
        "Plotter.py",
        "Sampler.py",
        "Solution.py",
        "Solver.py",
        "types.py",
        "utils.py",
        "py.typed",
    ]
    
    print("\n📦 Checking library modules in src/pybasin/...")
    missing = []
    for module in expected_modules:
        full_path = src_pybasin / module
        if full_path.exists():
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} (MISSING)")
            missing.append(module)
    
    return len(missing) == 0


def check_installation():
    """Check if the package is installed."""
    print("\n📦 Checking package installation...")
    try:
        import pybasin
        print(f"  ✅ pybasin is installed (version: {getattr(pybasin, '__version__', 'unknown')})")
        return True
    except ImportError:
        print("  ❌ pybasin is not installed")
        print("     Run: uv pip install -e .")
        return False


def print_summary(checks_passed):
    """Print final summary."""
    print("\n" + "="*60)
    if all(checks_passed):
        print("✨ Migration completed successfully! ✨")
        print("\nNext steps:")
        print("  1. Update case study imports to use 'from pybasin import ...'")
        print("  2. Run tests: pytest")
        print("  3. Build docs: mkdocs serve")
        print("  4. Run case studies: ./scripts/run_all_case_studies.sh")
    else:
        print("⚠️  Migration incomplete - some checks failed")
        print("\nPlease review the missing items above.")
    print("="*60)


def main():
    """Run all checks."""
    print("🔍 pyBasin Project Structure Verification")
    print("="*60)
    
    checks = [
        check_directory_structure(),
        check_key_files(),
        check_library_modules(),
        check_installation(),
    ]
    
    print_summary(checks)
    
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
