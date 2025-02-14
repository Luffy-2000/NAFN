import os
import torch
import numpy as np
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict
import pytorch_lightning as pl


def collect_rng_states(include_cuda: bool = True) -> Dict[str, Any]:
    """
    Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python.
    """
    states = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": python_get_rng_state(),
    }
    if include_cuda:
        states["torch.cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    """
    Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process.
    """
    torch.set_rng_state(rng_state_dict["torch"])
    # torch.cuda rng_state is only included since v1.8.
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))
    

def reset_seed() -> None:
    """Reset the seed to the value that :func:`lightning.fabric.utilities.seed.seed_everything` previously set.

    If :func:`lightning.fabric.utilities.seed.seed_everything` is unused, this function will do nothing.
    """
    import pytorch_lightning as pl
    
    seed = os.environ.get("PL_GLOBAL_SEED", None)
    if seed is None:
        return
    pl.seed_everything(int(seed))
    

def seed_everything(seed):
    pl.seed_everything(seed)