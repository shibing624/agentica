# Contributing

We are happy to accept your contributions to make this repo better and more awesome! To avoid unnecessary work on either
side, please stick to the following process:

1. Check if there is already [an issue](https://github.com/shibing624/agentica/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs!
3. If we decide your concern needs code changes, we would be happy to accept a pull request. Please consider the
commit guidelines below.

## Running the tests

```sh
pip install -e ".[dev]" -e "./gateway[all]"
python -m pytest tests -q                       # serial: ~2.5 min
python -m pytest tests -q -n 8 --dist loadfile  # parallel (pytest-xdist): ~35 s
```

`-n` is a local convenience. Bare CI installs the package without `[dev]`,
where an unknown `-n` aborts collection, so it must stay out of `addopts`.
Before opening a PR also run `python scripts/check_bare_ci_imports.py`.
