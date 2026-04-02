# TurboQuant Release Checklist

## Correctness

- [ ] All TurboQuant tests are green (`core`, `keras`, `calibration`, `benchmark`).
- [ ] No backward-compatibility break in serialized encoding payloads.
- [ ] SavedModel export/load round-trip passes for representative models.

## Performance

- [ ] Benchmark suite runs on fixed hardware profile and stores JSON artifacts.
- [ ] Real-model benchmark report generated with multi-seed settings.
- [ ] Regression check confirms no unexpected drift/compression regressions.

## Documentation

- [ ] `README.md` reflects current CLI/tooling and APIs.
- [ ] `REPRODUCIBILITY.md` includes exact command lines used for release numbers.
- [ ] `KNOWN_LIMITATIONS.md` and `ROADMAP.md` updated.

## Release Metadata

- [ ] Release notes summarize behavior changes and migration risks.
- [ ] Commit history is linear, dated, and free of unrelated modifications.
