"""
DISCO - Dynamic Instruction Simulation for Computer Organization
simulator/instructions.py

Defines the simplified instruction set used by DISCO along with each
instruction's simulated clock-cycle cost. This module also parses and
validates a line of assembly-style source code into a structured
``Instruction`` object that the CPU simulator can execute.

IMPORTANT: The cycle costs below are SIMULATION ASSUMPTIONS used for this
educational COA model. They are not claims about the exact timing of any
real commercial processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Instruction set definition
# ---------------------------------------------------------------------------

# Simulated cycle cost per instruction (SIMULATION ASSUMPTION).
INSTRUCTION_CYCLES: Dict[str, int] = {
    "LOAD": 2,   # read a value from memory into a register
    "STORE": 2,  # write a register value into memory
    "ADD": 1,    # ALU add
    "SUB": 1,    # ALU subtract
    "MUL": 2,    # ALU multiply (assumed to cost more than add/sub)
    "MOV": 1,    # move an immediate/register value into a register
}

# Instructions that touch memory (used to drive the memory + cache models).
MEMORY_INSTRUCTIONS = {"LOAD", "STORE"}

# Instructions that perform an ALU (arithmetic) operation.
ALU_INSTRUCTIONS = {"ADD", "SUB", "MUL"}


class InstructionError(ValueError):
    """Raised when an instruction is invalid or malformed."""


@dataclass
class Instruction:
    """A single decoded instruction.

    Attributes:
        opcode:   The mnemonic, e.g. ``"ADD"``.
        operands: The raw operand tokens as written by the user.
        cycles:   The simulated clock-cycle cost for this instruction.
        raw:      The original source line (useful for the execution trace).
    """

    opcode: str
    operands: List[str] = field(default_factory=list)
    cycles: int = 0
    raw: str = ""

    @property
    def is_memory(self) -> bool:
        """True if this instruction accesses memory (LOAD/STORE)."""
        return self.opcode in MEMORY_INSTRUCTIONS

    @property
    def is_alu(self) -> bool:
        """True if this instruction uses the ALU (ADD/SUB/MUL)."""
        return self.opcode in ALU_INSTRUCTIONS


def supported_opcodes() -> List[str]:
    """Return the list of supported instruction mnemonics."""
    return list(INSTRUCTION_CYCLES.keys())


def _is_register(token: str) -> bool:
    """A register looks like R0, R1, ... R<n>."""
    return len(token) >= 2 and token[0].upper() == "R" and token[1:].isdigit()


def _is_integer(token: str) -> bool:
    """True if the token is a (possibly signed) integer literal."""
    try:
        int(token)
        return True
    except ValueError:
        return False


def _split_line(line: str) -> tuple[str, List[str]]:
    """Strip comments/labels and split a line into (opcode, operands)."""
    # Remove inline comments starting with ';' or '#'.
    for comment_char in (";", "#"):
        if comment_char in line:
            line = line.split(comment_char, 1)[0]
    line = line.strip()
    if not line:
        return "", []

    # First whitespace-separated token is the opcode; the rest are operands.
    parts = line.split(None, 1)
    opcode = parts[0].upper()
    operand_str = parts[1] if len(parts) > 1 else ""
    operands = [tok.strip() for tok in operand_str.split(",") if tok.strip()]
    return opcode, operands


def _validate_operands(opcode: str, operands: List[str]) -> None:
    """Validate operand shape for a given opcode. Raises InstructionError."""
    if opcode == "LOAD":
        # LOAD Rdest, <address>
        if len(operands) != 2 or not _is_register(operands[0]) or not _is_integer(operands[1]):
            raise InstructionError(
                f"LOAD expects 'LOAD Rdest, address' (got: {operands})"
            )
    elif opcode == "STORE":
        # STORE Rsrc, <address>
        if len(operands) != 2 or not _is_register(operands[0]) or not _is_integer(operands[1]):
            raise InstructionError(
                f"STORE expects 'STORE Rsrc, address' (got: {operands})"
            )
    elif opcode in ("ADD", "SUB", "MUL"):
        # OP Rdest, Rsrc1, Rsrc2
        if len(operands) != 3 or not all(_is_register(op) for op in operands):
            raise InstructionError(
                f"{opcode} expects '{opcode} Rdest, Rsrc1, Rsrc2' (got: {operands})"
            )
    elif opcode == "MOV":
        # MOV Rdest, <register|immediate>
        if len(operands) != 2 or not _is_register(operands[0]):
            raise InstructionError(
                f"MOV expects 'MOV Rdest, value' (got: {operands})"
            )
        if not (_is_register(operands[1]) or _is_integer(operands[1])):
            raise InstructionError(
                f"MOV source must be a register or integer (got: {operands[1]})"
            )


def parse_line(line: str) -> Instruction:
    """Parse and validate a single source line into an ``Instruction``.

    Accepted (simplified) syntax examples::

        LOAD  R1, 10        ; load memory[10] into R1
        STORE R4, 100       ; store R4 into memory[100]
        ADD   R3, R1, R2    ; R3 = R1 + R2
        MUL   R4, R3, R2    ; R4 = R3 * R2
        MOV   R1, 5         ; R1 = 5

    Returns:
        A populated ``Instruction``.

    Raises:
        InstructionError: if the opcode is unknown or operands are malformed.
    """
    raw = line.strip()
    opcode, operands = _split_line(line)
    if not opcode:
        raise InstructionError("Empty instruction line.")

    if opcode not in INSTRUCTION_CYCLES:
        raise InstructionError(
            f"Unknown instruction '{opcode}'. "
            f"Supported: {', '.join(supported_opcodes())}"
        )

    _validate_operands(opcode, operands)

    return Instruction(
        opcode=opcode,
        operands=operands,
        cycles=INSTRUCTION_CYCLES[opcode],
        raw=raw,
    )


def parse_program(source: str) -> List[Instruction]:
    """Parse a multi-line program into a list of ``Instruction`` objects.

    Blank lines and comment-only lines are skipped. Errors include the line
    number so users can locate the problem easily.

    Raises:
        InstructionError: if the program is empty or any line is invalid.
    """
    instructions: List[Instruction] = []
    for line_no, line in enumerate(source.splitlines(), start=1):
        stripped, _ = _split_line(line)
        if not stripped:
            continue  # skip blank / comment-only lines
        try:
            instructions.append(parse_line(line))
        except InstructionError as exc:
            raise InstructionError(f"Line {line_no}: {exc}") from exc

    if not instructions:
        raise InstructionError("Workload is empty. Please provide at least one instruction.")

    return instructions
