# Pytest's `pythonpath = ["src"]` setting in pyproject.toml makes
# `asl_live` importable from tests with no manual sys.path setup.
# Conftest exists so pytest treats `tests/` as a rootdir reliably.
