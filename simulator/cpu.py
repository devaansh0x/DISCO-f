"""
DISCO - Dynamic Instruction Simulation for Computer Organization
simulator/cpu.py

A simplified, educational CPU model. It demonstrates the classic
instruction cycle:

    FETCH -> DECODE -> EXECUTE -> MEMORY ACCESS (if needed) -> WRITE BACK

The CPU keeps the state a student expects to see in a COA course:
    - a program counter (PC)
    - a small register file
    - a running clock-cycle count
    - an instruction count
    - a human-readable execution trace

It works together with the Memory and Cache models so that LOAD/STORE
instructions exercise the cache and contribute to the performance metrics.

This is NOT a real ISA (x86/ARM). It is a teaching model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .cache import Cache
from .instructions import Instruction, parse_program
from .memory import Memory, MemoryError_


class CPUError(Exception):
    """Raised for runtime errors during CPU simulation (e.g. bad register)."""


@dataclass
class TraceEntry:
    """One row of the execution trace, capturing each pipeline stage."""

    step: int
    pc: int
    instruction: str
    stages: str          # e.g. "FETCH -> DECODE -> EXECUTE -> WRITEBACK"
    cache_event: str     # "HIT", "MISS" or "-" for non-memory instructions
    cycles: int          # cycles consumed by this instruction
    total_cycles: int    # running total after this instruction
    note: str            # short description of what happened


@dataclass
class CPU:
    """A simplified single-cycle-per-instruction educational CPU.

    Args:
        num_registers: Size of the register file (R0..R<n-1>).
        memory:        Memory model to use (created if not supplied).
        cache:         Cache model to use (created if not supplied).
    """

    num_registers: int = 8
    memory: Memory = field(default_factory=Memory)
    cache: Cache = field(default_factory=Cache)

    def __post_init__(self) -> None:
        if self.num_registers <= 0:
            raise ValueError("CPU must have at least one register.")
        # Register file R0..R<n-1>, all initialised to 0.
        self.registers: Dict[str, int] = {
            f"R{i}": 0 for i in range(self.num_registers)
        }
        self.pc: int = 0                       # program counter
        self.clock_cycles: int = 0             # total simulated cycles
        self.instruction_count: int = 0        # instructions executed
        self.trace: List[TraceEntry] = []      # execution trace

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reg(self, name: str) -> int:
        """Read a register value, validating that it exists."""
        if name not in self.registers:
            raise CPUError(
                f"Invalid register '{name}'. Valid: R0..R{self.num_registers - 1}."
            )
        return self.registers[name]

    def _set_reg(self, name: str, value: int) -> None:
        """Write a register value, validating that it exists."""
        if name not in self.registers:
            raise CPUError(
                f"Invalid register '{name}'. Valid: R0..R{self.num_registers - 1}."
            )
        self.registers[name] = int(value)

    def _resolve_value(self, token: str) -> int:
        """Resolve a token that is either a register name or an integer."""
        if token in self.registers:
            return self.registers[token]
        try:
            return int(token)
        except ValueError as exc:
            raise CPUError(f"Cannot resolve operand '{token}'.") from exc

    # ------------------------------------------------------------------
    # Execution of a single instruction (EXECUTE / MEM / WRITEBACK stages)
    # ------------------------------------------------------------------
    def _execute(self, instr: Instruction) -> tuple[str, str, str]:
        """Execute one instruction.

        Returns:
            (stages, cache_event, note) describing what happened, for the trace.
        """
        op = instr.opcode

        if op == "MOV":
            dest, src = instr.operands
            value = self._resolve_value(src)
            self._set_reg(dest, value)
            return ("FETCH -> DECODE -> EXECUTE -> WRITEBACK",
                    "-", f"{dest} = {value}")

        if op in ("ADD", "SUB", "MUL"):
            dest, s1, s2 = instr.operands
            a, b = self._reg(s1), self._reg(s2)
            if op == "ADD":
                result = a + b
            elif op == "SUB":
                result = a - b
            else:  # MUL
                result = a * b
            self._set_reg(dest, result)
            return ("FETCH -> DECODE -> EXECUTE(ALU) -> WRITEBACK",
                    "-", f"{dest} = {a} {op} {b} = {result}")

        if op == "LOAD":
            dest, addr_token = instr.operands
            address = int(addr_token)
            hit = self.cache.access(address)      # cache is consulted first
            try:
                value = self.memory.read(address)  # then main memory
            except MemoryError_ as exc:
                raise CPUError(str(exc)) from exc
            self._set_reg(dest, value)
            return ("FETCH -> DECODE -> EXECUTE -> MEMORY -> WRITEBACK",
                    "HIT" if hit else "MISS",
                    f"{dest} = MEM[{address}] = {value}")

        if op == "STORE":
            src, addr_token = instr.operands
            address = int(addr_token)
            hit = self.cache.access(address)
            value = self._reg(src)
            try:
                self.memory.write(address, value)
            except MemoryError_ as exc:
                raise CPUError(str(exc)) from exc
            return ("FETCH -> DECODE -> EXECUTE -> MEMORY",
                    "HIT" if hit else "MISS",
                    f"MEM[{address}] = {src} = {value}")

        # Should never reach here because instructions are validated on parse.
        raise CPUError(f"Unsupported opcode at runtime: {op}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, program: List[Instruction]) -> None:
        """Run a parsed program to completion, recording the trace."""
        for instr in program:
            # FETCH + DECODE are implicit; here we simulate their effect.
            stages, cache_event, note = self._execute(instr)

            self.clock_cycles += instr.cycles
            self.instruction_count += 1
            self.pc += 1

            self.trace.append(
                TraceEntry(
                    step=self.instruction_count,
                    pc=self.pc - 1,
                    instruction=instr.raw,
                    stages=stages,
                    cache_event=cache_event,
                    cycles=instr.cycles,
                    total_cycles=self.clock_cycles,
                    note=note,
                )
            )

    def run_source(self, source: str) -> None:
        """Convenience: parse a source string then run it."""
        self.run(parse_program(source))

    def trace_as_dicts(self) -> List[Dict[str, object]]:
        """Return the execution trace as a list of plain dicts (for tables)."""
        return [
            {
                "Step": t.step,
                "PC": t.pc,
                "Instruction": t.instruction,
                "Stages": t.stages,
                "Cache": t.cache_event,
                "Cycles": t.cycles,
                "Total Cycles": t.total_cycles,
                "Result": t.note,
            }
            for t in self.trace
        ]

    def register_snapshot(self) -> Dict[str, int]:
        """Return a copy of the current register file (for display)."""
        return dict(self.registers)
