"""
DISCO - Dynamic Instruction Simulation for Computer Organization
simulator/memory.py

A simplified, deterministic main-memory (RAM) model. It exists to
demonstrate the relationship between CPU execution and memory access.
This is NOT an operating-system memory subsystem; there is no paging,
protection, virtual addressing, or allocation.
"""

from __future__ import annotations

from typing import Dict


class MemoryError_(Exception):
    """Raised when a memory address is invalid (out of range)."""


class Memory:
    """A simple word-addressable RAM.

    Addresses run from 0 to ``size - 1``. Any location that has never been
    written reads back as 0, which keeps the model deterministic.
    """

    def __init__(self, size: int = 1024, preseed: bool = True) -> None:
        """Create a memory of ``size`` words.

        Args:
            size: Number of addressable words. Must be positive.
            preseed: If True (default), each address is pre-initialised so that
                MEM[address] == address. This is a deterministic convenience
                that makes LOAD instructions read meaningful (non-zero) values
                for demonstration, e.g. ``LOAD R1, 10`` yields R1 = 10. It does
                not change the memory model; STORE still overwrites cells and
                addresses never touched still read back their seeded value.
        """
        if size <= 0:
            raise ValueError("Memory size must be a positive integer.")
        self.size: int = size
        self._preseed: bool = preseed
        # Sparse storage: only written addresses are kept. Unwritten locations
        # read back as their address (if preseeded) or 0 otherwise.
        self._cells: Dict[int, int] = {}

    def _check_address(self, address: int) -> None:
        """Validate that ``address`` is within range."""
        if not isinstance(address, int):
            raise MemoryError_(f"Memory address must be an integer (got {address!r}).")
        if address < 0 or address >= self.size:
            raise MemoryError_(
                f"Invalid memory address {address}. Valid range is 0..{self.size - 1}."
            )

    def read(self, address: int) -> int:
        """Read a word from memory.

        A written location returns its stored value. An unwritten location
        returns its seeded value (equal to the address) when ``preseed`` is
        enabled, otherwise 0.
        """
        self._check_address(address)
        default = address if self._preseed else 0
        return self._cells.get(address, default)

    def write(self, address: int, value: int) -> None:
        """Write ``value`` to ``address`` in memory."""
        self._check_address(address)
        self._cells[address] = int(value)

    def snapshot(self) -> Dict[int, int]:
        """Return a copy of the currently written memory cells (for display)."""
        return dict(self._cells)
