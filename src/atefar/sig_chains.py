from distutils.command import build
import dspy
from atefar.dspy_utils import build_cumulative_module, DspyStep
from atefar.fields import *

full_paper_chain = build_cumulative_module(
    inputs = [paper_content],
    steps = [
        DspyStep([title, abstract_plus]),
        DspyStep([quantitative_results_json]),
        DspyStep([core_ideas_json]),
        DspyStep([metrics_json]),
        DspyStep([hw_agnostic_metrics_json]),
        DspyStep([baseline_methods_json]),
        DspyStep([experimental_methods_json]),
        DspyStep([method_metric_results_json]),
        DspyStep([task_candidates_json]),
        DspyStep([task_prerequisites_json]),
        DspyStep([task_eval_instructions_json]),
        DspyStep([task_eval_baseline_implementation_json]),
        DspyStep([task_eval_correctness_scoring_json]),
        DspyStep([task_eval_metric_scoring_json]),
        DspyStep([task_eval_combined_scoring_json]),
        DspyStep([task_setup_script]),
    ]
)

paper_context_chain = build_cumulative_module(
    inputs = [paper_content],
    steps = [
        DspyStep([title, abstract_plus]),
        DspyStep([quantitative_results_json]),
        DspyStep([core_ideas_json]),
        DspyStep([metrics_json]),
        DspyStep([hw_agnostic_metrics_json]),
        DspyStep([baseline_methods_json]),
        DspyStep([experimental_methods_json]),
        DspyStep([method_metric_results_json]),
    ]
)