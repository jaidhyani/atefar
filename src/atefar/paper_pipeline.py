from dotenv import load_dotenv, dotenv_values
import dspy
import dspy.teleprompt
from atefar.pdf_utils import extract_text_from_pdf
from pydantic import BaseModel, Field
from typing import List
import json
import baml_py as baml
from typing import Callable
from anthropic.types import Message
from datetime import datetime
from textwrap import dedent


# Load environment variables
# we use load_dotenv to set variables directly in the environment
# and dotenv_values to get the variables in a dictionary
# so we can reference them in the code without the linter complaining
load_dotenv()
env = dotenv_values()


# Event hacks
class Event:
    def __init__(self, name: str):
        self.name = name
        self.listeners: dict[str, Callable] = {}
    
    def add_listener(self, name: str, listener: Callable, overwrite: bool=False):
        if name in self.listeners and not overwrite:
            raise ValueError(f"Listener with name {name} already exists")
        self.listeners[name] = listener
    
    def remove_listener(self, name) -> Callable:
        return self.listeners.pop(name)
    
    def fire(self, *args, **kwargs):
        for listener in self.listeners.values():
            listener(*args, **kwargs)
    

class HookedClaude(dspy.Claude):
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: str | None = None, api_base: str | None = None, **kwargs):
        super().__init__(model, api_key, api_base, **kwargs)
        self.request_event = Event("request")
        self.response_event = Event("response")


    def basic_request(self, prompt: str, **kwargs):
        self.request_event.fire(prompt, **kwargs)
        response = super().basic_request(prompt, **kwargs)
        self.response_event.fire(response, **kwargs)
        return response


logs = []

def message_to_str(msg: Message) -> str:
    blocks = [
        block.text for block in msg.content if block.type == "text"
    ]
    return "\n".join(blocks)

def log_message(msg: Message, prefix: str="Message: ", max_len: int=50):
    text = message_to_str(msg)
    log_str(text, prefix, max_len)

def log_str(text: str, prefix: str, max_len: int=50):
    if len(text) > max_len:
        short_text = text[:max_len] + "..."
    else:
        short_text = text
    logs.append({prefix: text})
    print(f"{prefix}: {short_text}")

        

def configure_lm():
    """Configure and return a language model."""
    lm = HookedClaude("claude-3-5-sonnet-20240620", api_key=env["ANTHROPIC_API_KEY"])
    # "claude-3-sonnet-20240229"
    lm.request_event.add_listener("log", lambda prompt, **kwargs: log_str(prompt, "Request"))
    lm.response_event.add_listener("log", lambda response, **kwargs: log_message(response, "Response"))
    dspy.configure(lm=lm)
    return lm

lm = configure_lm()

class TaskCandidate(BaseModel):
    """Eval task candidate extracted from a research paper to be used as the basis for evaluating research/engineering capabilities

    A good task candidate:
    - describes a task that an agent can implement
    - typically a programming task, e.g. modifying a 'baseline' implementation
    - which reflects or mirrors actual tasks undertaken to produce the research paper
    - and requires research and/or engineering expertise to implement correctly
    - for which it is relatively straightforward to write an objective scoring function which aassigns scores to task implementations
    - ideally, it should be much easier to produce an implementation scoring function (given access to the paper) than to implement the task itself (without access to the paper)
    - the ultimate goal is to evaluate the research/engineering capabilities of an agent stronger than the agent writing the task specification
    """
    name: str = Field(description="Name of the task")
    description: str = Field(description="Description of the task")
    baseline: str = Field(description="Short description of baseline implementation, if any, that will be given to implementers")
    skills: list[str] = Field(description="Comma-separate list of skills required to implement the task")
    relevant_paper_text: str = Field(description="Text from the paper that is relevant to the task")
    scoring_feasibility: int = Field(
        description=dedent(
            """
            On a scale of 1-10, how feasible is it to write a python function to objectively score an 
            implementation of this task, verifying that key functionality is implemented as specified?
            """
        )
    )
    llm_tractability: float = Field(description="Probability in (0, 1) that a frontier LLM can generate a correct implementation of this task")
    expert_tractability: float = Field(description="Probability in (0, 1) that a human subject matter expert can generate a correct implementation of this task")
    layman_tractability: float = Field(description="Probability in (0, 1) that a layman can generate a correct implementation of this task")
    scoring_llm_tractability: float = Field(description="Probability in (0, 1) that a frontier LLM can generate a correct scoring function for this task")
    asset_prerequisites: list[str] = Field(description="List of assets which should be made available to implementers or which are required to implement scoring")

    def __str__(self):
        return self.model_dump_json()

class TaskCandidates(BaseModel):
    tasks: List[TaskCandidate] = Field(description="Tasks extracted from the paper")

class PaperTaskCandidatesSignature(dspy.Signature):
    paper_text: str = dspy.InputField(desc="Full text from research paper PDF")
    guidance: str = dspy.InputField(desc="Guidance for extracting task candidates", optional=True)
    task_candidates: TaskCandidates = dspy.OutputField(desc="JSON list of task candidate dicts with keys 'name', 'description', and 'relevant_paper_text'")

class SelfCritiqueSignature(dspy.Signature):
    candidate_input: str = dspy.InputField()
    candidate_output: str = dspy.InputField()
    requirements: str = dspy.InputField()
    attempt_num: int = dspy.InputField()
    previous_critiques: str = dspy.InputField()
    self_critique: str = dspy.OutputField(desc="Self-critique of how well the candidate output satisfies the requirements given the input")
    should_retry: bool = dspy.OutputField(desc="Should we attempt to generate the output again taking this critique into account? True/False")

class TaskCriterion(BaseModel):
    """
    Objective, programatically verifiable criterion for evaluating a task
    implementation. Criteria may call a function and use outputs and observed
    side effects to determine if the criterion is satisfied, but will
    otherwise not be able to make any assumptions about how the implementation
    works - it is a black box.
    Criteria should never be subjective or require examining the actual code in 
    the implementation. It must be strictly empirical.
    """
    criterion: str = Field()
    importance: float = Field(description="Float in [0, 100] indicating how important satisfying this criterion is to a successful implementation?")
    

def __str__(self):
        return f"Criterion: {self.criterion}\nImportance: {self.importance}"
class TaskRubric(BaseModel):
    rubric: list[TaskCriterion] = Field(description=dedent("""
        List of criteria for evaluating task implementation; importance should sum to 100.
    """))

    def __str__(self):
        return "\n".join([str(criterion) for criterion in self.rubric])

class TaskRubricSignature(dspy.Signature):
    task_candidate: str = dspy.InputField()
    guidance: str = dspy.InputField()
    task_rubric: TaskRubric = dspy.OutputField()


class ScorableJudge(dspy.Signature):
    task: str = dspy.InputField()
    rubric: str = dspy.InputField()
    scorable: bool = dspy.OutputField(desc=dedent("""
        Will it be possible to write a python function to objectively score
        an implementation of this task, verifying that key functionality 
        is implemented as specified? The scoring function will only have access to the
        function implementing the task, and we cannot make any assumptions about
        how the task is implemented by the agent. We only want to validate correctness.
    """), prefix="Scorable[True/False]:")
    justification: str = dspy.OutputField(desc="Justification for scorable judgement")


class TaskImplementation(BaseModel):
    name: str = Field(description="Name of the task")
    instructions: str = Field(description="Instructions to be given to the agent for implementing the task. Instructions should be comprehensive, precise and not refer to the source paper.")
    baseline: str = Field(description="Python code to be given to the agent as a starting point for implementing the task")

    implementation: str
    scoring_function: str
    score: float


scorable_judge = dspy.ChainOfThought(ScorableJudge)
task_candidates_predictor = dspy.TypedPredictor(PaperTaskCandidatesSignature)
task_candidates_self_critique_predictor = dspy.TypedPredictor(SelfCritiqueSignature)
generic_self_critique_predictor = dspy.TypedPredictor(SelfCritiqueSignature)
task_rubrics_predictor = dspy.TypedPredictor(TaskRubricSignature)


class TaskGenerationPipeline(dspy.Module):
    def __init__(self):
        self.task_candidates_predictor = dspy.TypedPredictor(PaperTaskCandidatesSignature)
        self.task_candidates_self_critique_predictor = dspy.TypedPredictor(SelfCritiqueSignature)
        self.task_rubrics_predictor = dspy.TypedPredictor(TaskRubricSignature)
        self.scorable_judge = dspy.ChainOfThought(ScorableJudge)
    
    def forward(self, paper_text: str):
        candidate_critiques = []
        candidate_iterations = 0
        critique = ""
        while candidate_iterations < 3:
            task_candidate_response = self.task_candidates_predictor(paper_text=paper_text, guidance=critique)
            task_candidate_critique_response = self.task_candidates_self_critique_predictor(
                candidate_input=paper_text,
                candidate_output=str(task_candidate_response.task_candidates),
                requirements="Extract all promising task candidates from paper. " + TaskCandidate.__doc__,
                attempt_num=candidate_iterations,
                previous_critiques="\n".join(candidate_critiques)
            )
            candidate_iterations += 1
            critique = task_candidate_critique_response.self_critique
            candidate_critiques.append(critique)

            if not task_candidate_critique_response.should_retry:
                break

        all_task_analysis = {}
        for task in task_candidate_response.task_candidates.tasks:
            assert isinstance(task, TaskCandidate)
            task_analysis = task.model_dump()
            analysis_iteration = 0
            task_critiques = [""]
            while analysis_iteration < 3:
                task_rubric_response = self.task_rubrics_predictor(task_candidate=str(task), guidance=task_critiques[-1])
                scorable_response = self.scorable_judge(task=str(task), rubric=str(task_rubric_response.task_rubric))
                analysis_iteration += 1
                self_critique = generic_self_critique_predictor(
                    candidate_input=str(task),
                    candidate_output=str(task_rubric_response),
                    requirements="Define a rubric for programming task evaluation, where criteria satisfy " + TaskCriterion.__doc__,
                    attempt_num=analysis_iteration,
                    previous_critiques="\n".join(candidate_critiques)
                )
                if not self_critique.should_retry:
                    break
                task_critiques.append(self_critique.self_critique)
            task_analysis["rubric"] = task_rubric_response.task_rubric.model_dump()
            scorable_response = self.scorable_judge(task=str(task), rubric=str(task_rubric_response.task_rubric))
            task_analysis["scorable"] = scorable_response.scorable
            task_analysis["justification"] = scorable_response.justification
            print(task.name, scorable_response)
            all_task_analysis[task.name] = task_analysis
        return dspy.Prediction(task_analysis = all_task_analysis)

paper_text = extract_text_from_pdf("papers/94cifar.pdf")
pipeline = TaskGenerationPipeline()
results = pipeline(paper_text=paper_text)

now = datetime.now().isoformat()
with open(f"{now}_logs.json", "w") as f:
    json.dump(logs, f, indent=2)
with open(f"{now}_task_analysis.json", "w") as f:
    json.dump(results.task_analysis, f, indent=2)
