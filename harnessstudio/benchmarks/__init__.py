"""HarnessStudio-owned subclasses of vla-eval benchmarks (spec 16.11).

Two kinds are admitted:
- sensor subclasses that publish extra *sensor* data a real robot has (depth, camera calibration), e.g.
  ``libero_rgbd``;
- protocol subclasses that reproduce a published evaluation protocol on top of a vla-eval benchmark (case
  definition / reset, action post-processing, success metrics), e.g. ``simpler_seeded`` (EXT simpler-seeded).

Neither kind may give the policy or a method simulator object state, reward or success: observations stay the
benchmark's public sensor data. Reference them by import string in a vla-eval config and mount this checkout into the
benchmark container (e.g. volume <repo>:/hs and env PYTHONPATH=/hs). Importing this package must only need vla-eval
and numpy (other dependencies are imported lazily inside methods).
"""
