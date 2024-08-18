import dspy
from dataclasses import dataclass
from dspy.primitives.program import Module
from typing import Any, Optional

@dataclass
class GenericField:
    name: str
    desc: str

def as_input(field: GenericField):
    return dspy.InputField(name=field.name, desc=field.desc)

def as_output(field: GenericField):
    return dspy.OutputField(name=field.name, desc=field.desc)

@dataclass
class DspyStep:
    outputs: list[GenericField]
    module: Module = dspy.ChainOfThought
    module_kwargs: Optional[dict[str, Any]] = None
    # TODO: implement
    inputs: list[GenericField] = None  # can be used to override default include-all behavior
    asserts: list[dspy.Assert] = None
    suggests: list[dspy.Suggest] = None

@dataclass
class DspyChain:
    inputs: list[GenericField]
    steps: list[DspyStep]

def build_cumulative_module(inputs = list[GenericField], steps = list[DspyStep]):
    input_sigs = inputs.copy()
    modules = []
    for step in steps:
        class StepSig(dspy.Signature):
            pass
        for i in input_sigs:
            StepSig = StepSig.append(i.name, dspy.InputField(desc=i.desc))
        for o in step.outputs:
            StepSig = StepSig.append(o.name, dspy.OutputField(desc=o.desc))
        input_sigs.extend(step.outputs)
        step_module = step.module(StepSig, **(step.module_kwargs or {}))
        modules.append(step_module)

    class CumulativeModule(dspy.Module):
        def __init__(self, **kwargs):
            super().__init__()

            self.cumulative_inputs = kwargs.copy()
            self.steps = modules            
            self.results = {}
        
        def forward(self):
            for step in self.steps:
                print(f"Running step {step}")
                print(f"  Inputs: {self.cumulative_inputs}")
                result = step(**self.cumulative_inputs)
                print(f"  Result: {[k for k in result.keys()]}")
                for value_name in result.keys():
                    self.cumulative_inputs[value_name] = result[value_name]
                self.results['_'.join([k for k in result.keys() if k != "rationale"])] = result
            return self.results
    return CumulativeModule