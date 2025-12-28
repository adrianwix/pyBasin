# Installation

## Prerequisites

- Python 3.11 or higher
- pip or uv package manager

## Install from PyPI

Once published, you'll be able to install pyBasin using pip:

```bash
pip install pybasin
```

## Install from Source

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/adrianwix/pyBSTAB.git
cd pyBasinWorkspace

# Install with UV
uv add -e .

# Or install with all optional dependencies
uv add -e ".[all]"
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/adrianwix/pyBSTAB.git
cd pyBasinWorkspace

# Install in editable mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

## Optional Dependencies

### Development Tools

Install testing and linting tools:

```bash
uv add -e ".[dev]"
```

Includes:

- pytest
- pytest-cov
- mypy
- ruff
- black

### Documentation

Install documentation building tools:

```bash
uv add -e ".[docs]"
```

Includes:

- mkdocs-material
- mkdocstrings
- mkdocs-jupyter

### Case Studies

Install additional dependencies for running case studies:

```bash
uv add -e ".[case-studies]"
```

Includes:

- jupyter
- openpyxl
- notebook

## Verification

Verify your installation:

```python
import pybasin
print(pybasin.__version__)
```

## Next Steps

- Check out the [Quick Start](quickstart.md) guide
- Explore the [API Reference](../api/basin-stability-estimator.md)
- Run the [Case Studies](../case-studies/overview.md)
