# Legacy code: do not use for results

`Coverless_stego_baseline_invalid/` is the code that produced Exp01. Exp01 is invalid as a reproduction:

- The receiver ran no detector. `_verify_object_survival` decided accuracy with fixed thresholds.
- A "failed" object had its first bit flipped, forcing an error.
- The mapping used object instances, not the paper's label dictionary.
- Eq. 2 was modified, and the candidate search was limited to 58 images.
- The experiment switches in `config.py` did not change the algorithm.

It is kept only as a record. The Git tag `exp01-invalid` marks the last commit before the rewrite.
