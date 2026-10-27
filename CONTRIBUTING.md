# Contributing to causalmlp

This is a small research archive, not a packaged library. Contributions that make the experiment clearer, more reproducible, or more honestly measured are welcome.

## Workflow

1. Open an issue describing the change and the evidence it would produce.
2. Fork the repository and keep the change focused.
3. Make sure the modules still import and the CPU experiment still runs.
4. Open a pull request with the commands you ran and the output you observed.

## Evidence expectations

- Keep `exp(-cross-entropy)` and top-1 accuracy as separate metrics with their own names. Do not subtract one from the other.
- If you report a number, bind it to the code revision, data, seed, and metric definition.
- Do not add a benchmark, compatibility claim, or plot that was not produced by a run in this repository.
- Preserve negative and partial findings; they are part of the result.

## Manual checks

There is no CI. Before opening a pull request, run:

```bash
python causal_mlp.py
python train.py --device cpu --epochs 1
```

## License

By contributing, you agree that your contribution is licensed under the repository's GNU AGPL-3.0-only license. See [LICENSE](LICENSE).
