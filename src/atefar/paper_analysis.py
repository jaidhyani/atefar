import dspy

class PaperAnalysis(dspy.Signature):
    """Analyze a research paper and provide core information."""

    paper_content = dspy.InputField(desc="The full text content of the research paper")

    summary = dspy.OutputField(desc="A concise summary of the paper's main contributions")
    core_ideas = dspy.OutputField(desc="The core idea(s) of the paper")
    methods = dspy.OutputField(desc="Key methods or strategies proposed in the paper")
    metrics = dspy.OutputField(desc="Primary metrics or evaluation criteria used in the paper")
    requirements = dspy.OutputField(desc="Key requirements for implementing and evaluating the paper's methods")