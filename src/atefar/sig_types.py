import dspy
from dataclasses import dataclass
from dspy.primitives.program import Module
from typing import Any, Optional


@dataclass
class SigData:
    name: str
    desc: str

@dataclass
class SigStep:
    outputs: list[SigData]
    inputs: list[SigData] = None
    module: Module = dspy.ChainOfThought
    kwargs: Optional[dict[str, Any]] = None

@dataclass
class SigChain:
    inputs: list[SigData]
    steps: list[SigStep]