# Contributing

EpiCataNexus is currently maintained as a research-preview code release. Contributions
that improve documentation, tests, portability, preprocessing, or checkpoint loading
are welcome after the public repository is available.

## Development workflow

1. Create a focused branch.
2. Keep changes scoped and document scientific assumptions.
3. Do not commit datasets, checkpoints, pretrained feature caches, manuscript drafts,
   credentials, or local environment files.
4. Run the lightweight checks before opening a pull request:

```bash
pip install -e ".[dev]"
pytest
python scripts/smoke_test.py
```

Changes to tensor shapes, feature definitions, data splits, or evaluation metrics must
also update the corresponding documentation and tests.

## Reporting issues

Include the operating system, Python/PyTorch versions, CUDA version when relevant, the
exact command, checkpoint task and hash, input tensor shapes, and a minimal traceback.
Do not attach restricted datasets or unpublished model files to public issues.
