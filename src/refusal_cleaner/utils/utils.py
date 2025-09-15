# refusal_cleaner/utils.py
from typing import List, Tuple

def choose_batch_size(n: int) -> int:
    """Split into ~10 chunks, but keep chunk size >= 1000."""
    if n <= 0:
        return 0
    if n < 1000:
        return n
    return max(1000, n // 10)

def chunk_indices(n: int, bs: int) -> List[Tuple[int, int]]:
    return [(i, min(i + bs, n)) for i in range(0, n, bs)]
