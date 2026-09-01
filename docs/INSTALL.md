# Installation

## Reproducible environment

The research environment used Python 3.10, PyTorch 2.1.2, and CUDA 12.1. Start with:

```bash
conda env create -f environment.yml
conda activate epicatanexus
pip install -e .
```

## Mamba

`mamba-ssm` and `causal-conv1d` include compiled CUDA extensions. Install versions
that match the local PyTorch, CUDA, Python, and C++ ABI. Do not install the wheel files
from the original research workspace blindly; those files are intentionally excluded
from this repository.

After installing the matched packages, verify:

```bash
python -c "from mamba_ssm import Mamba; print('mamba-ssm available')"
```

## Lightweight development install

Documentation, result, metric, and non-Mamba component tests need only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/smoke_test.py
```

## External tools

Raw structure preprocessing uses fpocket 4.2.3. Structure retrieval may use PDB or
AlphaFold models, subject to their respective access terms. The release should record
the fpocket binary checksum and full commands before publication.

