"""Launcher: `.venv/bin/python -I serve.py --port 8141 --device cuda:0` (-I: the paths below are the only ones added)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pcd_perception.runtime import install_paths  # noqa: E402

install_paths()

from pcd_perception.server import main  # noqa: E402

if __name__ == "__main__":
    main()
