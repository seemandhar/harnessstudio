"""HarnessStudio-owned subclasses of vla-eval benchmarks that publish extra *sensor* data (spec 16.11).

They must only add what a real robot's sensors provide (depth, camera calibration), never simulator
object state. Reference them by import string in a vla-eval config and mount this checkout into the
benchmark container (e.g. volume <repo>:/hs and env PYTHONPATH=/hs). Importing this package must only
need vla-eval and numpy.
"""
