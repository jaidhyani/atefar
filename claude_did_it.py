# this doesn't work
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPRO
from dspy.evaluate import Evaluate
import re
from collections import Counter
import hashlib
import pickle
import ast
import os

from atefar import pdf_utils
pdf_text = pdf_utils.extract_text_from_pdf("papers/94cifar.pdf")


def configure_lm():
    """Configure and return a language model."""
    lm = dspy.Claude("claude-3-5-sonnet-20240620", api_key="***REMOVED***")
    dspy.configure(lm=lm)
    return lm

configure_lm()

# 1. Signatures


class PaperAnalysisSignature(dspy.Signature):
    """Analyze and structure a research paper."""
    paper_text = dspy.InputField()
    structured_paper = dspy.OutputField(desc="""
        Dict with keys:
        - title: string
        - abstract: string summarizing the paper
        - main_contributions: list of strings, each describing a key contribution
        - methodology: string describing the main methods/algorithms
        - evaluation: string describing evaluation metrics and datasets
        - results: dict with key findings and their corresponding metrics
        - additional_sections: dict with any other relevant sections
    """)

class ResearchContributionAnalysisSignature(dspy.Signature):
    """Analyze the key research contributions and their implications."""
    structured_paper = dspy.InputField()
    contribution_analysis = dspy.OutputField(desc="""
        Dict with keys:
        - core_algorithms: list of dicts, each containing:
            - name: string, name of the algorithm
            - description: string, brief description
            - novelty: string, what makes it novel
            - complexity: string, estimated computational complexity
        - technical_challenges: list of strings, key technical challenges addressed
        - datasets: list of dicts, each containing:
            - name: string, name of the dataset
            - description: string, brief description
            - usage: string, how it was used in the paper
        - evaluation_metrics: list of dicts, each containing:
            - name: string, name of the metric
            - description: string, what it measures and why it's relevant
        - key_results: list of dicts, each containing:
            - description: string, description of the result
            - significance: string, why this result is important
        - potential_extensions: list of strings, areas where the work could be extended
        - limitations: list of strings, current limitations of the approach
    """)

class TaskIdentificationSignature(dspy.Signature):
    """Identify challenging and relevant tasks based on the paper's contributions."""
    structured_paper = dspy.InputField()
    contribution_analysis = dspy.InputField()
    potential_tasks = dspy.OutputField(desc="""
        List of task dictionaries, each containing:
        - name: string, a concise name for the task
        - type: string, one of:
            - 'algorithm_implementation': Implement a core algorithm from the paper
            - 'result_reproduction': Reproduce a key result from the paper
            - 'model_extension': Extend the model to a new domain or dataset
            - 'efficiency_optimization': Optimize the algorithm for better performance
            - 'ablation_study': Conduct an ablation study on the model
            - 'comparative_analysis': Compare the paper's approach to other methods
        - description: string, detailed description of the task, including:
            - Clear goal
            - Specific steps or requirements
            - Expected challenges
        - relevance: string, explanation of why this task is important given the paper's contributions
        - required_expertise: list of strings, key areas of expertise needed (e.g., 'deep learning', 'optimization', 'NLP')
        - estimated_difficulty: string, one of ['challenging', 'very challenging', 'extremely challenging']
        - expected_outcome: string, what a successful completion of this task would demonstrate
        - evaluation_approach: string, high-level description of how submissions would be evaluated
    """)

class TaskFormulationSignature(dspy.Signature):
    """Formulate a task with detailed specifications."""
    task_info = dspy.InputField()
    formulated_task = dspy.OutputField(desc="""
        Dict with keys:
        - description: string, detailed task description including context from the paper
        - input_format: string, precise description of expected input (data format, ranges, etc.)
        - output_format: string, precise description of expected output
        - evaluation_criteria: list of strings, each describing a specific criterion
        - constraints: list of strings, any constraints or requirements for the implementation
        - resources: list of strings, suggested resources or references
        - estimated_time: string, estimated time to complete the task
    """)

class ScoringFunctionGenerationSignature(dspy.Signature):
    """Generate a Python scoring function for a task."""
    formulated_task = dspy.InputField()
    scoring_function = dspy.OutputField(desc="""
        Python code string for scoring function with the following structure:
        - Function name: score_task
        - Parameters: (submission, ground_truth, **kwargs)
        - Return: float between 0 and 1
        - Must include:
            - Input validation
            - Evaluation based on all criteria in formulated_task
            - Proper weighting of multiple criteria if applicable
            - Error handling
        - Docstring explaining the scoring methodology
    """)

# 2. Modules

class PaperAnalysis(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(PaperAnalysisSignature)

    def forward(self, paper_text):
        result = self.analyze(paper_text=paper_text)
        
        dspy.Suggest(
            self._validate_structure(result.structured_paper),
            "Ensure the paper structure captures the key elements of an AI/ML research paper."
        )
        
        return result

    def _validate_structure(self, structured_paper):
        essential_sections = {"title", "abstract", "main_contributions"}
        return len(set(structured_paper.keys()) & essential_sections) == len(essential_sections)

class ResearchContributionAnalysis(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ResearchContributionAnalysisSignature)

    def forward(self, structured_paper):
        result = self.analyze(structured_paper=structured_paper)
        
        dspy.Suggest(
            self._validate_analysis(result.contribution_analysis),
            "Ensure the analysis captures key elements needed for task creation."
        )
        
        return result

    def _validate_analysis(self, analysis):
        required_keys = {"core_algorithms", "technical_challenges", "evaluation_metrics", "key_results"}
        return all(key in analysis and analysis[key] for key in required_keys)

class TaskIdentification(dspy.Module):
    def __init__(self):
        super().__init__()
        self.identify = dspy.ProgramOfThought(TaskIdentificationSignature)

    def forward(self, structured_paper, contribution_analysis):
        result = self.identify(structured_paper=structured_paper, contribution_analysis=contribution_analysis)
        
        dspy.Suggest(
            self._validate_tasks(result.potential_tasks),
            "Ensure each task is challenging, relevant, and well-defined."
        )
        
        return result

    def _validate_tasks(self, tasks):
        def is_valid_task(task):
            return all([
                task['type'] in ['algorithm_implementation', 'result_reproduction', 'model_extension', 
                                 'efficiency_optimization', 'ablation_study', 'comparative_analysis'],
                len(task['description']) >= 100,  # Ensure detailed description
                len(task['required_expertise']) >= 2,  # Require multiple areas of expertise
                task['estimated_difficulty'] in ['challenging', 'very challenging', 'extremely challenging'],
                'evaluation_approach' in task
            ])
        return all(is_valid_task(task) for task in tasks)

class TaskFormulation(dspy.Module):
    def __init__(self):
        super().__init__()
        self.formulate = dspy.ChainOfThought(TaskFormulationSignature)

    def forward(self, task_info):
        result = self.formulate(task_info=task_info)
        
        dspy.Suggest(
            self._validate_formulation(result.formulated_task),
            "Ensure task formulation includes all required components."
        )
        
        return result

    def _validate_formulation(self, task):
        required_keys = {"description", "input_format", "output_format", "evaluation_criteria", "constraints"}
        return all(key in task for key in required_keys)

class ScoringFunctionGeneration(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.TypedChainOfThought(ScoringFunctionGenerationSignature)

    def forward(self, formulated_task):
        result = self.generate(formulated_task=formulated_task)
        
        dspy.Assert(
            self._validate_python_code(result.scoring_function),
            "Generated scoring function must be valid Python code."
        )
        
        return result

    def _validate_python_code(self, code_string):
        try:
            ast.parse(code_string)
            return True
        except SyntaxError:
            return False

# 3. Metrics

def paper_analysis_metric(example, pred):
    """Measure the quality of the paper analysis."""
    required_sections = {"title", "abstract", "main_contributions", "methodology", "evaluation", "results"}
    structured_paper = pred.structured_paper
    completeness = len(set(structured_paper.keys()) & required_sections) / len(required_sections)
    
    # Check for non-empty content in each section
    content_quality = sum(bool(structured_paper.get(section)) for section in required_sections) / len(required_sections)
    
    return (completeness + content_quality) / 2

def contribution_analysis_metric(example, pred):
    """Measure the quality of the contribution analysis."""
    required_keys = {"core_algorithms", "technical_challenges", "evaluation_metrics", "key_results"}
    contribution_analysis = pred.contribution_analysis
    completeness = len([k for k in required_keys if k in contribution_analysis and contribution_analysis[k]]) / len(required_keys)
    
    # Check for non-empty lists in each key
    content_quality = sum(bool(contribution_analysis.get(key)) for key in required_keys) / len(required_keys)
    
    return (completeness + content_quality) / 2

def task_relevance_metric(example, pred):
    """Measure how relevant the task is to cutting-edge AI/ML research."""
    relevance_keywords = set(['deep learning', 'neural network', 'machine learning', 'AI', 'optimization',
                              'algorithm', 'model', 'architecture', 'training', 'inference', 'performance'])
    task = pred.potential_tasks[0] if pred.potential_tasks else {}  # Assume we're evaluating the first task
    task_text = ' '.join([task.get('name', ''), task.get('description', ''), ' '.join(task.get('required_expertise', []))])
    keyword_count = sum(keyword in task_text.lower() for keyword in relevance_keywords)
    return min(keyword_count / 5, 1.0)  # Normalize to [0, 1]

def task_difficulty_metric(example, pred):
    """Assess if the task is appropriately challenging."""
    difficulty_scores = {'challenging': 0.6, 'very challenging': 0.8, 'extremely challenging': 1.0}
    task = pred.potential_tasks[0] if pred.potential_tasks else {}  # Assume we're evaluating the first task
    difficulty_score = difficulty_scores.get(task.get('estimated_difficulty', ''), 0)
    expertise_score = min(len(task.get('required_expertise', [])) / 5, 1.0)
    return (difficulty_score + expertise_score) / 2

def task_diversity_metric(example, pred):
    """Encourage a diverse set of task types."""
    task_types = [task.get('type', '') for task in pred.potential_tasks]
    unique_types = set(task_types)
    diversity_score = len(unique_types) / 6  # Assuming 6 possible task types
    balance_score = min(task_types.count(t) for t in unique_types) / max(task_types.count(t) for t in unique_types) if task_types else 0
    return (diversity_score + balance_score) / 2

def task_specificity_metric(example, pred):
    """Measure how well-defined and specific the task is."""
    task = pred.formulated_task
    has_clear_steps = 'steps' in task.get('description', '').lower() or 'procedure' in task.get('description', '').lower()
    has_expected_outcome = 'expected outcome' in task.get('description', '').lower() or 'goal' in task.get('description', '').lower()
    description_length = min(len(task.get('description', '').split()) / 100, 1.0)
    has_input_format = bool(task.get('input_format'))
    has_output_format = bool(task.get('output_format'))
    return (has_clear_steps + has_expected_outcome + description_length + has_input_format + has_output_format) / 5

def task_objectivity_metric(example, pred):
    """Assess how objectively evaluable the task is."""
    task = pred.formulated_task
    has_evaluation_criteria = 'evaluation_criteria' in task and len(task['evaluation_criteria']) > 0
    has_metrics = any('metric' in criterion.lower() for criterion in task.get('evaluation_criteria', []))
    has_quantitative_terms = any(term in ' '.join(task.get('evaluation_criteria', [])).lower() 
                                 for term in ['accuracy', 'error', 'performance', 'speed', 'efficiency'])
    return (has_evaluation_criteria + has_metrics + has_quantitative_terms) / 3

def scoring_function_complexity_metric(example, pred):
    """Assess the complexity and completeness of the scoring function."""
    scoring_function = pred.scoring_function
    required_elements = ["def score_task(", "submission", "ground_truth", "return", "try:", "except:"]
    basic_structure_score = sum(element in scoring_function for element in required_elements) / len(required_elements)
    
    # Check for more advanced elements
    has_multiple_criteria = len(re.findall(r'if.*?:', scoring_function)) > 1
    uses_external_library = any(lib in scoring_function for lib in ['numpy', 'scipy', 'sklearn'])
    has_weighted_scoring = 'weight' in scoring_function.lower()
    
    advanced_score = (has_multiple_criteria + uses_external_library + has_weighted_scoring) / 3
    
    return (basic_structure_score + advanced_score) / 2

def pipeline_metric(example, pred):
    """Evaluate the quality of the entire pipeline output."""
    # Assuming pred is a list of (task, scoring_function) tuples
    task_scores = [
        (task_relevance_metric(example, dspy.Prediction(potential_tasks=[task])) +
         task_difficulty_metric(example, dspy.Prediction(potential_tasks=[task])) +
         task_specificity_metric(example, dspy.Prediction(formulated_task=task)) +
         task_objectivity_metric(example, dspy.Prediction(formulated_task=task))) / 4
        for task, _ in pred
    ]
    
    scoring_scores = [scoring_function_complexity_metric(example, dspy.Prediction(scoring_function=score)) for _, score in pred]
    
    avg_task_quality = sum(task_scores) / len(task_scores) if task_scores else 0
    avg_scoring_quality = sum(scoring_scores) / len(scoring_scores) if scoring_scores else 0
    
    diversity_score = task_diversity_metric(example, dspy.Prediction(potential_tasks=[task for task, _ in pred]))
    
    return (0.4 * avg_task_quality + 0.3 * avg_scoring_quality + 0.3 * diversity_score)

# 4. Pipeline

class TaskExtractionPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.paper_analysis = PaperAnalysis()
        self.contribution_analysis = ResearchContributionAnalysis()
        self.task_identification = TaskIdentification()
        self.task_formulation = TaskFormulation()
        self.scoring_function_generation = ScoringFunctionGeneration()

    def forward(self, paper_text):
        structured_paper = self.paper_analysis(paper_text=paper_text).structured_paper
        contribution_analysis = self.contribution_analysis(structured_paper=structured_paper).contribution_analysis
        potential_tasks = self.task_identification(structured_paper=structured_paper, contribution_analysis=contribution_analysis).potential_tasks
        
        final_tasks = []
        for task_info in potential_tasks:
            formulated_task = self.task_formulation(task_info=task_info).formulated_task
            scoring_function = self.scoring_function_generation(formulated_task=formulated_task).scoring_function
            final_tasks.append((formulated_task, scoring_function))
        
        return final_tasks

# 5. Optimization Functions

# Caching decorator
def cache_result(func):
    cache = {}
    def wrapper(*args, **kwargs):
        key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

def save_pipeline(pipeline, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {filename}")

def load_pipeline(filename):
    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"Pipeline loaded from {filename}")
    return pipeline

    
    
def optimize_paper_analysis(module, trainset, max_demos=64, num_candidates=100):
    proper_trainset = [dspy.Example(paper_text=paper) for paper in trainset]
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=paper_analysis_metric,
        max_bootstrapped_demos=max_demos,
        num_candidate_programs=num_candidates
    )
    return optimizer.compile(student=module, trainset=proper_trainset)

def optimize_contribution_analysis(module, trainset, max_demos=64, num_candidates=100):
    # Assuming the input for this module is the output of paper_analysis
    proper_trainset = [dspy.Example(structured_paper=paper) for paper in trainset]
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=contribution_analysis_metric,
        max_bootstrapped_demos=max_demos,
        num_candidate_programs=num_candidates
    )
    return optimizer.compile(student=module, trainset=proper_trainset)

def optimize_task_identification(module, trainset, max_demos=64, num_candidates=100):
    # Assuming the input for this module includes both structured_paper and contribution_analysis
    proper_trainset = [dspy.Example(structured_paper=paper, contribution_analysis=analysis) 
                       for paper, analysis in trainset]
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=lambda ex, pred: (task_relevance_metric(ex, pred) + task_difficulty_metric(ex, pred) + task_diversity_metric(ex, pred)) / 3,
        max_bootstrapped_demos=max_demos,
        num_candidate_programs=num_candidates
    )
    return optimizer.compile(student=module, trainset=proper_trainset)

def optimize_task_formulation(module, trainset, max_demos=64, num_candidates=100):
    # Assuming the input for this module is task_info
    proper_trainset = [dspy.Example(task_info=task) for task in trainset]
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=lambda ex, pred: (task_specificity_metric(ex, pred) + task_objectivity_metric(ex, pred)) / 2,
        max_bootstrapped_demos=max_demos,
        num_candidate_programs=num_candidates
    )
    return optimizer.compile(student=module, trainset=proper_trainset)

def optimize_scoring_function_generation(module, trainset, max_demos=64, num_candidates=100):
    # Assuming the input for this module is formulated_task
    proper_trainset = [dspy.Example(formulated_task=task) for task in trainset]
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=scoring_function_complexity_metric,
        max_bootstrapped_demos=max_demos,
        num_candidate_programs=num_candidates
    )
    return optimizer.compile(student=module, trainset=proper_trainset)

def optimize_pipeline_extensive(pipeline, trainset, max_demos=64, num_candidates=100, num_iterations=10):
    best_pipeline = None
    best_score = float('-inf')

    for i in range(num_iterations):
        print(f"Pipeline optimization iteration {i+1}/{num_iterations}")
        optimizer = MIPRO(
            metric=pipeline_metric,
            max_bootstrapped_demos=max_demos,
            num_candidate_programs=num_candidates
        )
        optimized_pipeline = optimizer.compile(student=pipeline, trainset=trainset)
        
        evaluator = dspy.Evaluate(devset=trainset, metric=pipeline_metric)
        score = evaluator(optimized_pipeline)

        print(f"Iteration {i+1} score: {score}")

        if score > best_score:
            best_pipeline = optimized_pipeline
            best_score = score
            print(f"New best score: {best_score}")

    return best_pipeline

def optimize_modules(pipeline, train_papers):
    print("Optimizing paper analysis module...")
    pipeline.paper_analysis = optimize_paper_analysis(pipeline.paper_analysis, train_papers)
    
    # Generate structured papers for the next step
    structured_papers = [pipeline.paper_analysis(paper_text=paper).structured_paper for paper in train_papers]
    
    print("Optimizing contribution analysis module...")
    pipeline.contribution_analysis = optimize_contribution_analysis(pipeline.contribution_analysis, structured_papers)
    
    # Generate contribution analyses for the next step
    contribution_analyses = [pipeline.contribution_analysis(structured_paper=paper).contribution_analysis for paper in structured_papers]
    
    print("Optimizing task identification module...")
    pipeline.task_identification = optimize_task_identification(pipeline.task_identification, 
                                                                list(zip(structured_papers, contribution_analyses)))
    
    # Generate task infos for the next step
    task_infos = [task for paper, analysis in zip(structured_papers, contribution_analyses)
                  for task in pipeline.task_identification(structured_paper=paper, contribution_analysis=analysis).potential_tasks]
    
    print("Optimizing task formulation module...")
    pipeline.task_formulation = optimize_task_formulation(pipeline.task_formulation, task_infos)
    
    # Generate formulated tasks for the final step
    formulated_tasks = [pipeline.task_formulation(task_info=task).formulated_task for task in task_infos]
    
    print("Optimizing scoring function generation module...")
    pipeline.scoring_function_generation = optimize_scoring_function_generation(pipeline.scoring_function_generation, formulated_tasks)
    
    return pipeline


def main():
    papers = [
        pdf_text
    ]
    
    train_papers = papers
    test_papers = papers

    pipeline_file = 'optimized_pipeline.pkl'
    modules_file = 'optimized_modules.pkl'

    if os.path.exists(pipeline_file) and os.path.exists(modules_file):
        print("Loading previously optimized pipeline and modules...")
        optimized_pipeline = load_pipeline(pipeline_file)
        optimized_modules = load_pipeline(modules_file)
        # Reassign optimized modules to the pipeline
        optimized_pipeline.paper_analysis = optimized_modules.paper_analysis
        optimized_pipeline.contribution_analysis = optimized_modules.contribution_analysis
        optimized_pipeline.task_identification = optimized_modules.task_identification
        optimized_pipeline.task_formulation = optimized_modules.task_formulation
        optimized_pipeline.scoring_function_generation = optimized_modules.scoring_function_generation
    else:
        print("Creating and optimizing new pipeline...")
        pipeline = TaskExtractionPipeline()
        
        print("Optimizing individual modules...")
        optimized_modules = optimize_modules(pipeline, train_papers)
        
        print("Optimizing entire pipeline...")
        pipeline_trainset = [dspy.Example(paper_text=paper) for paper in train_papers]
        optimized_pipeline = optimize_pipeline_extensive(optimized_modules, pipeline_trainset)
        
        print("Saving optimized pipeline and modules...")
        save_pipeline(optimized_pipeline, pipeline_file)
        save_pipeline(optimized_modules, modules_file)

    print("Evaluating optimized pipeline...")
    test_examples = [dspy.Example(paper_text=paper) for paper in test_papers]
    evaluator = dspy.Evaluate(devset=test_examples, metric=pipeline_metric)
    results = evaluator(optimized_pipeline)
    print("Pipeline Evaluation Results:", results)
    
    print("Extracting tasks from papers...")
    for i, paper in enumerate(papers, 1):
        print(f"\nProcessing paper {i}:")
        extracted_tasks = optimized_pipeline(paper_text=paper)
        for j, (task, scoring_function) in enumerate(extracted_tasks, 1):
            print(f"\nTask {j}:")
            print(task)
            print("\nScoring Function:")
            print(scoring_function)

if __name__ == "__main__":
    main()


