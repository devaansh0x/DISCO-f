"""
DISCO - Dynamic Instruction Simulation for Computer Organization
simulator/cache.py

A simple, understandable cache simulation used to demonstrate how cache
behavior affects CPU performance. It uses a direct-mapped cache policy,
which is the easiest cache organization to explain in a viva:

    line index = address MOD number_of_lines

Each line stores which memory address (tag) it currently holds. If the
requested address maps to a line that already holds it -> HIT. Otherwise
-> MISS and the line is replaced with the new address.

This is an educational model, not a real multi-level cache hierarchy.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class Cache:
    """A direct-mapped cache tracking hits, misses and hit rate."""

    def __init__(self, num_lines: int = 8) -> None:
        """Create a cache.

        Args:
            num_lines: Number of cache lines. Must be positive. A larger cache
                       generally produces more hits for the same workload.
        """
        if num_lines <= 0:
            raise ValueError("Cache must have at least one line.")
        self.num_lines: int = num_lines
        # Each line holds the memory address currently cached, or None if empty.
        self._lines: List[Optional[int]] = [None] * num_lines
        self.hits: int = 0
        self.misses: int = 0

    def _index(self, address: int) -> int:
        """Map a memory address to a cache line (direct-mapped)."""
        return address % self.num_lines

    def access(self, address: int) -> bool:
        """Access ``address`` through the cache.

        Returns:
            True on a cache hit, False on a cache miss. On a miss the line is
            loaded with the requested address (replacing whatever was there).
        """
        index = self._index(address)
        if self._lines[index] == address:
            self.hits += 1
            return True
        # Miss: bring the requested address into the cache line.
        self._lines[index] = address
        self.misses += 1
        return False

    @property
    def total_accesses(self) -> int:
        """Total number of cache accesses (hits + misses)."""
        return self.hits + self.misses

    def hit_rate(self) -> float:
        """Cache hit rate as a percentage (0..100).

        Returns 0.0 when there have been no accesses, avoiding division by zero.
        """
        if self.total_accesses == 0:
            return 0.0
        return (self.hits / self.total_accesses) * 100.0

    def stats(self) -> Dict[str, float]:
        """Return a structured summary of cache statistics."""
        return {
            "cache_accesses": self.total_accesses,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": round(self.hit_rate(), 2),
        }
