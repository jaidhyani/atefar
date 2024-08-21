import dspy
import dspy.teleprompt
from atefar.pdf_utils import extract_text_from_pdf
from pydantic import BaseModel, Field
from typing import List
import json
import baml_py as baml



def configure_lm():
    """Configure and return a language model."""
    # lm = dspy.Claude("claude-3-5-sonnet-20240620", api_key="***REMOVED***")
    lm = dspy.Claude("claude-3-sonnet-20240229", api_key="***REMOVED***")
    dspy.configure(lm=lm)
    return lm

lm = configure_lm()

class TaskCandidate(BaseModel):
    name: str = Field(description="Name of the task")
    description: str = Field(description="Description of the task")
    relevant_paper_text: str = Field(description="Text from the paper that is relevant to the task")

class TaskCandidates(BaseModel):
    tasks: List[TaskCandidate] = Field(description="Tasks extracted from the paper")

class PaperTasks(dspy.Signature):
    paper_text: str = dspy.InputField(desc="Full text from research paper PDF")
    task_candidate_list: TaskCandidates = dspy.OutputField()



class TaskJudge(dspy.Signature):
    task: TaskCandidate = dspy.InputField()
    scorable: bool = dspy.OutputField(desc="Will it be possible to write a function to objectively score a task implementation, verifying that key functionality is implemented?", prefix="Scorable[True/False]:")

judge = dspy.ChainOfThought(TaskJudge)

predictor = dspy.TypedPredictor(PaperTasks)

paper_text = extract_text_from_pdf("papers/94cifar.pdf")

try:
    prediction = predictor(paper_text=paper_text)
    for task in prediction.task_candidate_list.tasks[:5]:
        task.scorable = judge(paper_text=paper_text, task=task)
        print(task.name, task.scorable)
except Exception as e:
    with open("history.json", "w") as f:
        json.dump(str(lm.history), f)
    print("Error:", e)
    print("History saved to history.json")
    raise e



def pdf_to_tasks(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    tasks = dspy.text_to_tasks(text)
    return tasks