"""Run a script with a GPU memory cap (sharing a GPU with probe runs): python capfrac.py 0.25 extract.py --configs ..."""
import runpy
import sys

import torch

torch.cuda.set_per_process_memory_fraction(float(sys.argv[1]))
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
