"""Small, deterministic CPU fixtures using only torch and pytest."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

# Load the repo example directly without requiring examples to be an installed
# distribution. This also lets pytest run with the documented PYTHONPATH=src.
_path = Path(__file__).resolve().parents[1] / "examples" / "synthetic.py"
_spec = importlib.util.spec_from_file_location("sala_synthetic_example", _path)
assert _spec is not None and _spec.loader is not None
_example = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _example
_spec.loader.exec_module(_example)


@pytest.fixture
def synthetic_inputs():
    return _example.make_synthetic_inputs()


@pytest.fixture(scope="session", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
