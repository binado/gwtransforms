# my_backend.py
try:
    import jax.numpy as np

    backend = "jax"
except ImportError:
    try:
        import autograd.numpy as np

        backend = "autograd"
    except ImportError:
        import numpy as np

        backend = "numpy"

HAS_JAX = backend == "jax"
HAS_AUTOGRAD = backend == "autograd"
HAS_AUTODIFF = HAS_JAX or HAS_AUTOGRAD

__all__ = ["np", "backend", "HAS_JAX", "HAS_AUTOGRAD", "HAS_AUTODIFF"]
