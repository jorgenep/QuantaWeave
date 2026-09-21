# Contributing To QuantaWeave

1. Create a focused branch or pull request for each change.
2. Keep generated datasets, checkpoints, virtual environments, and benchmark outputs out of commits.
3. Run the checks below before opening a pull request:

```bash
.axolotl-venv/bin/python -m py_compile src/*.py
/usr/bin/bash -n scripts/*.sh
.axolotl-venv/bin/python -m pytest -q tests
```

For model changes, also run a short CPU smoke test and record the device, parameter count, loss, and router metrics.

Please describe changes to routing, checkpoint formats, dataset schemas, or device backends in the pull request description.