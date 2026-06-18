import os
import random
import warnings

import numpy as np
import torch


DEFAULT_SEED = 42


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def set_seed(seed=DEFAULT_SEED, deterministic=True):
    deterministic = _env_flag("STREAMRING_DETERMINISTIC", deterministic) and deterministic

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False

    if deterministic:
        torch.use_deterministic_algorithms(True)


def set_runtime_threads():
    threads = os.environ.get("STREAMRING_NUM_THREADS")
    if not threads:
        return
    try:
        torch.set_num_threads(max(1, int(threads)))
    except ValueError:
        warnings.warn(f"Invalid STREAMRING_NUM_THREADS={threads!r}; keeping current thread count.")


    num_threads = int(os.environ.get("STREAMRING_NUM_THREADS", "1"))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        pass
