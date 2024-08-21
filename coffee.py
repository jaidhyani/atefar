import dspy
from pydantic import BaseModel, Field 
from typing import Literal, List
from dspy.functional import TypedPredictor
from dspy import Signature, InputField, OutputField

def configure_lm():
    """Configure and return a language model."""
    # lm = dspy.Claude("claude-3-5-sonnet-20240620", api_key="***REMOVED***")
    lm = dspy.Claude("claude-3-sonnet-20240229", api_key="***REMOVED***")
    dspy.configure(lm=lm)
    return lm

lm = configure_lm()

class AssertReason(BaseModel):
      assertion: str = Field()
      reason: str = Field()
      answer: List[Literal["A", "B", "C", "D"]] = Field()
    
class AssertReasonDSPy(Signature):
    """ Generate a list of assertions and reasons for the given context."""
    context: str = InputField()
    items: AssertReason = OutputField()


predictor = TypedPredictor(AssertReasonDSPy)
text = "Coffee One day around 850 CE, a goat herd named Kaldi observed that, after nibbling on some berries, his goats started acting abnormally. Kaldi tried them himself, and soon enough, he was just as hyper. This was humanity's first run-in with coffee— or so the story goes."
predictor(context=text)