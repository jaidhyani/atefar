import dspy
from atefar.sig_types import SigChain, SigStep
from atefar.sig_data import *

paper_chain_v5 = SigChain(
    inputs = [paper_content],
    steps = [
        SigStep([title, abstract_plus]),
        SigStep([quantitative_results_json]),
        SigStep([core_ideas_json]),
        SigStep([metrics_json]),
        SigStep([hw_agnostic_metrics_json]),
        SigStep([baseline_methods_json]),
        SigStep([experimental_methods_json]),
        SigStep([method_metric_results_json]),
        SigStep([task_candidates_json]),
        SigStep([task_prerequisites_json]),
        SigStep([task_eval_instructions_json]),
        SigStep([task_eval_baseline_implementation_json], module=dspy.ProgramOfThought),
        SigStep([task_eval_correctness_scoring_json], module=dspy.ProgramOfThought),
        SigStep([task_eval_metric_scoring_json], module=dspy.ProgramOfThought),
        SigStep([task_eval_combined_scoring_json], module=dspy.ProgramOfThought),
        SigStep([task_setup_script]),
    ]
)