import numpy as np
from numpy.typing import NDArray

try:
    from joblib import Memory
    # Create tmpdir for caching if it doesn't exist
    import tempfile
    tmpdir = tempfile.gettempdir()
    memory = Memory(location=tmpdir, mmap_mode='r', verbose=0)  # Use memory-mapped files for caching
except ImportError:
    memory = None
    print("Warning: joblib not installed. Caching of compute_statistics will be disabled.")


@memory.cache if memory else lambda func: func
def compute_statistics(feats: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute the mean and covariance of the features.
    """
    mu = np.mean(feats, axis=0)
    sig = np.cov(feats, rowvar=False)
    return mu, sig