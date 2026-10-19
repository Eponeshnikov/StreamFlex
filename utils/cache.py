"""File pickle cache (cache_result) and cached parquet reads."""

import hashlib
import os
import pickle

import streamlit as st

from utils.storage import resolve_cache_dir


def generate_cache_filename(cache_dir, func, *args, **kwargs):
    """
    Generate a unique cache filename based on the function name, arguments, and keyword arguments.

    This function takes a function object, positional arguments, and keyword arguments as input.
    It combines the function name, arguments, and keyword arguments into a tuple, serializes the tuple,
    and computes the SHA-256 hash of the serialized data. The hash is then used to generate a unique
    filename with a '.pickle' extension. The filename is returned as a string.

    ``cache_dir`` is only joined onto the result, never hashed, so the same
    call keys to the same filename on whichever disk the cache lives.

    Parameters
    ----------
    cache_dir (str): Directory the cache file belongs in.
    func (function): The function object for which the cache filename is being generated.
    *args (tuple): Positional arguments passed to the function.
    **kwargs (dict): Keyword arguments passed to the function.

    Returns:
    str: A unique cache filename based on the function name, arguments, and keyword arguments.
    """
    combined_data = (func.__name__, args, kwargs)
    serialized_data = pickle.dumps(combined_data)
    hash_object = hashlib.sha256(serialized_data)
    filename = f"{func.__name__}^" + hash_object.hexdigest() + ".pickle"
    return os.path.join(cache_dir, filename)


def parquet_signature(path: str) -> tuple:
    """Sizes and mtimes of a parquet file, or of a folder of part files.

    A cheap cache key that changes when the data does: a rebuilt dataset or
    a re-saved run keys differently without anyone clearing a cache.
    """
    if os.path.isdir(path):
        entries = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".parquet")
        )
    else:
        entries = [path]
    return tuple(
        (entry, os.stat(entry).st_size, os.stat(entry).st_mtime_ns)
        for entry in entries
    )


@st.cache_resource(max_entries=2, ttl="30m", show_spinner="Reading parquet…")
def _read_parquet_cached(path: str, signature: tuple):
    del signature  # only part of the cache key
    import polars as pl

    return pl.read_parquet(path)


def read_parquet_cached(path: str):
    """``pl.read_parquet(path)``, read once per file version, not per rerun.

    For pages that load a whole table and then slice it by widgets: every
    interaction used to re-read the file, which for a ResultsSaver run is
    gigabytes. ``cache_resource`` returns the same frame without a copy
    (polars frames are immutable), keyed on :func:`parquet_signature`; the
    TTL lets an idle page give the memory back.
    """
    return _read_parquet_cached(path, parquet_signature(path))


def cache_result(reset=False, cache_dir=None):
    """
    A decorator function that caches the results of a function and stores them in a cache file.

    Parameters
    ----------
    reset (bool): If True, the cache file will be deleted and the function will be executed again.
                If False (default), the function will attempt to load the result from the cache file.
    cache_dir (str): Where the cache files live. None resolves at call time
                from the scratch root (see :func:`resolve_cache_dir`), so a
                plugin can hand its own choice down to a loky worker through
                the per-config payload.

    Returns:
    function: The decorated function, which will either return the cached result or execute the function
            and store the result in the cache file.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            folder = cache_dir or resolve_cache_dir()
            cache_file = generate_cache_filename(folder, func, *args, **kwargs)
            if not reset:
                try:
                    with open(cache_file, "rb") as file:
                        cached_data = pickle.load(file)
                    cached_result = cached_data
                    return cached_result
                except (OSError, pickle.PickleError, EOFError):
                    pass
            result = func(*args, **kwargs)
            cached_data = result
            os.makedirs(folder, exist_ok=True)
            with open(cache_file, "wb") as file:
                pickle.dump(cached_data, file)

            return result

        return wrapper

    return decorator
