import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TIMEX_DATA_ROOT", PROJECT_ROOT / "dataset")).expanduser().resolve()
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"


def dataset_path(name):
    """Return a dataset directory below ``TIMEX_DATA_ROOT``."""
    return DATA_ROOT / name


model_types = ['tsimple']
exp_methods = ['fit', 'dyna', 'winit', 'tsr', 'sgt+grad']
