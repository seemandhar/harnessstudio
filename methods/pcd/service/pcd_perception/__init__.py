"""PCD counterfactual-image service (PCD-Mask GroundingDINO path), HarnessStudio v2.

Independent re-derivation of the ICLR code path of Koorye/PCD @ 31b027e (no licence; nothing copied, every rule
cites file:line in the docstrings), using Apache-2.0 third-party code vendored blob-identically under
`../third_party/` (see `../NOTICE`). Frozen models only; nothing is fitted.
"""

SERVICE_VERSION = "pcd-perception-0.1.0"
UPSTREAM_PIN = "Koorye/PCD@31b027ecf2a8b9a4e496aaa9478fb951e3d9270a"
