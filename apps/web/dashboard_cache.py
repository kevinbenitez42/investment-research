"""Small cache helpers for the notebook dashboard."""


def store_bounded(cache, key, value, *, limit):
    """Store a value, evicting only the oldest insertion when at capacity.

    Replacing an existing entry refreshes its insertion order. Callers own
    their cache keys and copy mutable figures before modifying them.
    """
    if limit < 1:
        raise ValueError("Cache limit must be positive.")
    cache.pop(key, None)
    while len(cache) >= limit:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value
