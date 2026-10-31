# roboharness

RoboHarness: A Memory-Augmented Policy Harness for Vision-Language-Action Model Robustness via In-Context Adaptation (arXiv:2603.24060) as a HarnessStudio method.

- What is reproduced: TODO
- What is substituted (and why the fidelity label): TODO

```bash
uv pip install --python .venv/bin/python -e methods/roboharness
hs verify roboharness --tier 0 && hs verify roboharness
hs eval roboharness --config <vla-eval config> --policy-url ws://... --contract libero --profile <profile> --out runs/roboharness
```
