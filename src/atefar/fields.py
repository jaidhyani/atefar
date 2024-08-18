from dataclasses import dataclass

import dspy

from atefar.dspy_utils import GenericField


paper_content = GenericField(
    "paper_content", 
    """
    The full text content of an AI research paper. Our eventual goal is to produce tasks the correspond to replicating parts of this paper 
    which can be used to evaluate the capabilities of researchers and developers. In other words, we want to identify one or more methods
    that the paper explores, as well as how the efficacy of those methods was measured (metrics). Then we want to develop a task wherein we will
    ask the subject to replicate some method explored in the paper in the form of a python function. For each task, we will eventually produce: 
    a description of the function to implement, the inputs to that function, the expected outputs from that function, (if applicable) a baseline
    implementation of the function to be given to the subject so that they can focus on implementing just the change we're interested in,
    and a scoring function for evaluating the correctness of a given function implementation.
    """
)

title = GenericField("title", "The title of the paper")

abstract_plus = GenericField(
    "abstract_plus", 
    """
    The abstract of the paper, plus optional additional high-level summaries to capture 
    interesting aspects of the paper not otherwise covered in the abstract
    """
)

quantitative_results_json = GenericField(
    "quantitative_results_json",
    """
    JSON list of quantitative results from the paper. 
    Example:
    [
        {{
            "units": "test_set_accuracy",
            "value": 0.89,
            "description": "Accuracy on ImageNet validation set",
            "method": "8 layer convnet with skip connections",
            "notes": "Trained for 5 epochs",
            "comparison": {{"baseline": 0.82, "improvement": "+7%"}} 
        }}
    ]

    Not every result will have every field, nulls are allowed and occasionally expected.
    """
)
core_ideas_json = GenericField(
    "core_ideas_json", 
    """
    JSON dict of ideas or approaches that the paper then goes on to demonstrate enable objective improvements according to certain metrics.
    
    The core idea in "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012) might be:
    {"deep_cnn": "Deep Convolutional neural networks can be used to classify images with high accuracy"}

    Example for "Attention Is All You Need" (Vaswani et al., 2017):
    {
        "attention_is_all_you_need": "Attention mechanisms can be used to model long-range dependencies in sequences",
        "attn_seq_to_seq": "Attention can be highly effective for sequence-to-sequence tasks like machine translation and other NLP tasks",
        "self_attn": "Self-attention mechanisms can be used to model dependencies between different parts of the same sequence",
        "multiheaded": "Multi-head attention mechanisms can be used to model different types of dependencies in parallel",
        "positional_encodings": "Positional encodings can be used to provide information about the position of tokens in a sequence",
        "transformer": "The Transformer architecture can be used to combine these ideas into a highly effective model for sequence-to-sequence tasks"
    }
    """
)

metrics_json = GenericField(
    "metrics_json", 
    """
    An enumerated list of the key metrics used to measure results in the paper. Response should be a JSON list of objects, where each object represents a metric. For example:
    [
        {
            "name": "flops",
            "description": "Number of floating point operations required to train a model to a certain level of accuracy",
            "unit": "GigaFLOPS",
            "measurement_details": "Measured using NVIDIA's nvprof tool during training on a V100 GPU",
            "justification": "FLOPs are a common measure of computational complexity and indicate how much raw computation an algorithm requires"
        },
        {
            "name": "test_set_accuracy",
            "description": "Percentage of correct predictions on a held-out test set",
            "unit": "Percentage",
            "measurement_details": "Evaluated on the ImageNet validation set after training for 90 epochs",
            "justification": "Accuracy is a common measure of model performance and generalization"
        }
    ]
    """
)

hw_agnostic_metrics_json = GenericField(
    "hw_agnostic_metrics_json",
    """
    JSON dict of metrics that correspond to the paper's metrics, but are not hardware-specific. Example:
    [
        {
            "name": "flops_per_training_step",
            "description": "Number of floating point operations per training step",
            "corresponding_hw_metric": "Training time per step",
            "unit": "GigaFLOPs",
            "equivalence_justification": "Higher FLOPs per step generally lead to longer training times, but this metric is generally independent of hardware"
        },
        {
            "name": "iterations_to_99_percent_accuracy",
            "description": "Number of training iterations required to reach 99% test set accuracy",
            "corresponding_hw_metric": "Training time to reach 99% accuracy",
            "unit": "Iterations",
            "equivalence_justification": "More iterations generally lead to longer training times, but this metric is independent of hardware"
        }
    ]
    """
)

baseline_methods_json = GenericField(
    "baseline_methods_json", 
    """
    JSON dict of baseline approaches that the paper's methods are compared against and/or iterating upon, 
    as well as a list of experimental methods that improve on or are compared to this baseline in the paper.
    Example:
    {
        "standard_transformer": {
            "description": "A sequence-to-sequence model based on self-attention mechanisms",
            "key_components": [
                "Multi-head self-attention layers",
                "Feed-forward neural networks",
                "Layer normalization",
                "Positional encodings"
            ],
            "architecture_details": {
                "encoder_layers": 6,
                "decoder_layers": 6,
                "attention_heads": 8,
                "embedding_dim": 512
            },
            "training_details": {
                "optimizer": "Adam",
                "learning_rate": "Warmup over 4000 steps, then decay",
                "batch_size": 128
            },
            "target_metrics": ["BLEU score", "Inference speed", "Model size"],
            "experimental_methods": ["sparse_attention_transformer", "dynamic_attention_patterns"]
        }
    }
    """
)

experimental_methods_json = GenericField(
    "experimental_methods_json", 
    """
    JSON list of dicts: key methods or strategies proposed in the paper to optimize the target metrics. 
    Example:
    [
        {
            "name": "sparse_attention_transformer",
            "baseline": "standard_transformer",
            "modifications": [
                "Replace full attention with sparse attention patterns",
                "Implement fixed or learned attention patterns"
            ],
            "expected_improvements": [
                "Reduced computational complexity from O(n²) to O(n log n)",
                "Better handling of long sequences"
            ],
            "target_metrics": ["Inference speed", "Memory usage", "Performance on long-sequence tasks"]
        }
    ]
    """
)

method_metric_results_json = GenericField(
    "method_metric_results", 
    """
    JSON dict of metric results that were reported in the paper for baseline and experimental methods.
    For example:
    {
        "standard_transformer": {
            "BLEU_score": 28.4,
            "inference_time": 100,
            "model_size": 65
        },
        "sparse_attention_transformer": {
            "BLEU_score": 28.2,
            "inference_time": 80,
            "model_size": 66
        }
    }
    """
)


task_candidates_json = GenericField(
    "task_candidates_json",
    """
    JSON dict of of potential tasks that could be used to evaluate an engineer or AI agent's ability to implement 
    these methods. This will be used later to build an instruction to the engineer, a python function to be implemented, and a 
    scoring function to evaluate the correctness of the implementation.

    Example:
    {
        {
            "name": "implement_sparse_attention",
            "description": "Implement a sparse attention mechanism in the MultiHeadAttention class",
            "corresponding_method": "Sparse Attention Transformer",
            "inputs": [
                {
                    "name": "input_tensor",
                    "type": "torch.Tensor",
                    "shape": "(batch_size, seq_length, d_model)"
                }
            ],
            "outputs": [
                {
                    "name": "output_tensor",
                    "type": "torch.Tensor",
                    "shape": "(batch_size, seq_length, d_model)"
                }
            ],
            "skills_tested": ["PyTorch", "Attention mechanisms", "Transformer architecture"],
            "assets_provided": ["baseline_implementation", "input_data"],
            "minimum_hardware_requirements": "1x A100 GPU or equivalent; 16GB RAM",
            "evaluation_criteria": [
                "Correctness: Compare attention patterns with expected sparsity",
                "Performance: Measure speed and memory usage improvements",
            ],
            "provided_baseline": "a standard transformer",
            "instructions_short": "Modify the provided MultiHeadAttention class to implement a sparse attention mechanism.",
            "time_to_complete": 4.5,
            "difficulty": 4,
            "feasibility": 5,
            "research_ability": 3
        }
    }
    """
)

task_prerequisites_json = GenericField(
    "task_prerequisites_json",
    """
    JSON dict of prerequisites that an engineer should have in order to complete each task.
    This may include libraries, datasets, pretrained models or weights. Later, this information will be used to
    write a setup script that will provide these prerequisites to the engineer.
    For example:
    {
        "implement_sparse_attention": ["PyTorch", "Transformers library", "ImageNet dataset"]
    }
    """
)

task_eval_instructions_json = GenericField(
    "task_eval_instructions_json", 
    """
    For each task candidate, write detailed instructions that will be provided to the engineer to complete the task. 
    These should expand on the short instructions in the task candidate, providing more context and guidance on how to complete the task.
    Instructions may refer to a input data, a baseline implementation, or other assets provided to the engineer.
    These instructions will be referred to when implementing any baseline to be provided to the engineer.
    Instructions should be explicit and specific enough that the engineer can complete the task without further guidance,
    and their solution can be objectively evaluated without human supervision. 
    Response should be a JSON object where keys are task names and values are the instructions. For example:
    {
        "implement_sparse_attention": "Modify the provided MultiHeadAttention class to implement a sparse attention mechanism. Your implementation should:\n1. Replace the full attention matrix with a sparse attention pattern (e.g., local + global attention)\n2. Ensure the sparse attention matrix is properly masked and normalized\n3. Maintain compatibility with the rest of the Transformer architecture\n\nInputs and outputs should remain the same as in the original implementation. Focus on modifying the 'forward' method to incorporate sparse attention."
    }
    """
)

task_eval_baseline_implementation_json = GenericField(
    "task_eval_baseline_implementation_json", 
    """
    JSON dict of baseline implementations for each task.
    The baseline should provide the basic structure of the function to be implemented, and enable the engineer 
    to focus on implementing the specific change we're interested in. 
    Not every task will have a baseline implementation. 
    Some baselines may be very simple (e.g. because the task requires implementing a new function from scratch), 
    while others may be more complex (e.g. because the task requires modifying an existing function).
    When possible, perfer continuous scoring functions that can be used to compare implementations.
    Higher scores should indicate better performance. The baseline implemention should score 0.0.
    Response should be a JSON object where keys are task candidate names and values are the baseline implementations as strings. For example:
    {
        "implement_sparse_attention": "```python\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        self.num_heads = num_heads\n        self.d_model = d_model\n        \n        assert d_model % self.num_heads == 0\n        \n        self.depth = d_model // self.num_heads\n        \n        self.wq = nn.Linear(d_model, d_model)\n        self.wk = nn.Linear(d_model, d_model)\n        self.wv = nn.Linear(d_model, d_model)\n        \n        self.dense = nn.Linear(d_model, d_model)\n        \n    def split_heads(self, x, batch_size):\n        x = x.view(batch_size, -1, self.num_heads, self.depth)\n        return x.permute(0, 2, 1, 3)\n    \n    def forward(self, q, k, v, mask=None):\n        batch_size = q.size(0)\n        \n        q = self.wq(q)\n        k = self.wk(k)\n        v = self.wv(v)\n        \n        q = self.split_heads(q, batch_size)\n        k = self.split_heads(k, batch_size)\n        v = self.split_heads(v, batch_size)\n        \n        scaled_attention_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)\n        \n        if mask is not None:\n            scaled_attention_logits += (mask * -1e9)\n        \n        attention_weights = F.softmax(scaled_attention_logits, dim=-1)\n        \n        output = torch.matmul(attention_weights, v)\n        \n        output = output.permute(0, 2, 1, 3).contiguous()\n        output = output.view(batch_size, -1, self.d_model)\n        output = self.dense(output)\n        \n        return output\n```",
    }
    """
)

task_eval_correctness_scoring_json = GenericField(
    "task_eval_correctness_scoring_json", 
    """
    JSON dict of task name to python functions named "score_solution_correctness" that will be used to score correctness of the implementation. 
    Higher scores indicate better performance. The baseline implemention should score 0.0.
    should take either one or two arguments: the implementation to be scored, 
    and (optionally) the baseline implementation.

    {
        "implement_sparse_attention": "```python\ndef score_sparse_attention_implementation_(modified_transformer, baseline_transformer):\n    score = 0.0\n    \n    # Test preserved functionality\n    short_input = torch.randn(32, 50, 512)\n    if torch.allclose(modified_transformer(short_input), baseline_transformer(short_input), atol=1e-5):\n        score += 0.3\n    \n    # Test improvement\n    long_input = torch.randn(32, 1000, 512)\n    modified_output = modified_transformer(long_input)\n    \n    # Check for sparse attention pattern\n    attention_weights = modified_transformer.encoder.layers[0].self_attn.attn_weights\n    if attention_weights.float().to_dense().count_nonzero() / attention_weights.numel() < 0.2:\n        score += 0.4\n    \n    # Check for improved efficiency\n    start_time = time.time()\n    modified_transformer(long_input)\n    modified_time = time.time() - start_time\n    \n    start_time = time.time()\n    baseline_transformer(long_input)\n    baseline_time = time.time() - start_time\n    \n    if modified_time < 0.8 * baseline_time:\n        score += 0.3\n    \n    return score\n```"
    }
    """
)


task_eval_metric_scoring_json = GenericField(
    "task_eval_metric_scoring_json", 
    """
    JSON dict of python functions that will be used to score the metric results of the implementation for each task.
    This should measure how well the implementation improves the target metrics, and will be used to compare the submitted implementation
    to corresponding results in the source paper, with the expectation of finding similar improvements.
    When possible, perfer continuous scoring functions that can be used to compare implementations.
    Higher scores should indicate better performance.
    The scoring function should take either one or two arguments: the implementation to be scored, 
    and (optionally) the baseline implementation. Any data used for evaluation should be hardcoded
    into the function, NOT passed as an argument. The function may use common libraries like huggingface to
    download and load data, but should not require any additional arguments to be passed in.
    
    Any data loading should be explicitly defined in the function, and should not rely on external data sources or other functions 
    (other than common libraries). e.g. use `datasets.load_dataset` to load data, but do not use a custom function to load data.

    {
        "implement_sparse_attention": "```python\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom datasets import load_dataset\nfrom transformers import AutoTokenizer\nimport time\nimport math\n\ndef score_sparse_attention_metric(sparse_attention_class, baseline_attention_class=None):\n    # Load a sample of the WikiText-2 dataset\n    dataset = load_dataset(\"wikitext\", \"wikitext-2-raw-v1\", split=\"test[:1000]\")\n    tokenizer = AutoTokenizer.from_pretrained(\"gpt2\")\n    \n    # Tokenize the dataset\n    def tokenize_function(examples):\n        return tokenizer(examples[\"text\"], padding=\"max_length\", truncation=True, max_length=128)\n    \n    tokenized_dataset = dataset.map(tokenize_function, batched=True)\n    \n    # Create data loader\n    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=32, shuffle=False)\n    \n    # Initialize models\n    d_model = 512\n    num_heads = 8\n    sparse_attention = sparse_attention_class(d_model, num_heads)\n    baseline_attention = baseline_attention_class(d_model, num_heads) if baseline_attention_class else None\n    \n    def evaluate_attention(attention_module):\n        total_time = 0\n        total_memory = 0\n        num_batches = 0\n        \n        for batch in dataloader:\n            input_ids = batch['input_ids']\n            attention_mask = batch['attention_mask']\n            \n            # Convert to appropriate tensor type\n            input_ids = input_ids.long()\n            attention_mask = attention_mask.float()\n            \n            # Create a sample input tensor\n            input_tensor = torch.randn(input_ids.shape[0], input_ids.shape[1], d_model)\n            \n            # Measure time\n            start_time = time.time()\n            with torch.no_grad():\n                _ = attention_module(input_tensor, input_tensor, input_tensor, mask=attention_mask)\n            end_time = time.time()\n            \n            total_time += end_time - start_time\n            \n            # Measure memory\n            torch.cuda.empty_cache()\n            torch.cuda.reset_peak_memory_stats()\n            with torch.no_grad():\n                _ = attention_module(input_tensor, input_tensor, input_tensor, mask=attention_mask)\n            total_memory += torch.cuda.max_memory_allocated()\n            \n            num_batches += 1\n        \n        avg_time = total_time / num_batches\n        avg_memory = total_memory / num_batches\n        \n        return avg_time, avg_memory\n    \n    sparse_time, sparse_memory = evaluate_attention(sparse_attention)\n    \n    if baseline_attention:\n        baseline_time, baseline_memory = evaluate_attention(baseline_attention)\n        time_improvement = (baseline_time - sparse_time) / baseline_time\n        memory_improvement = (baseline_memory - sparse_memory) / baseline_memory\n    else:\n        # If no baseline is provided, we'll compare against expected improvements\n        time_improvement = 1 - (sparse_time / 0.1)  # Assuming 0.1s is a good baseline\n        memory_improvement = 1 - (sparse_memory / 1e9)  # Assuming 1GB is a good baseline\n    \n    # Calculate sparsity\n    with torch.no_grad():\n        input_tensor = torch.randn(32, 128, d_model)\n        attention_weights = sparse_attention(input_tensor, input_tensor, input_tensor).squeeze()\n    sparsity = 1 - (torch.count_nonzero(attention_weights) / attention_weights.numel())\n    \n    # Score calculation\n    time_score = min(max(time_improvement, 0), 1)  # Clamp between 0 and 1\n    memory_score = min(max(memory_improvement, 0), 1)  # Clamp between 0 and 1\n    sparsity_score = min(max(sparsity, 0), 1)  # Clamp between 0 and 1\n    \n    final_score = (0.4 * time_score + 0.4 * memory_score + 0.2 * sparsity_score) * 100\n    \n    return final_score```"
    }
    """
)

task_eval_combined_scoring_json = GenericField(
    "task_eval_combined_scoring_json",
    """
    JSON dict of python functions that will be used to score the overall performance of the implementation for each task.
    This score indicate how well the engineer performed overall on the task. This may combine correctness and metric scoring functions,
    or use other criteria to evaluate the implementation. The ideal scoring function will have a low floor (easy to score above 0),
    high ceiling (score continues to scale with quality of implementation), and be continious.

    The scoring function should take exactly one argument, the implementation to be scored, and return a float.

    {
        "implement_sparse_attention": "```python\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom datasets import load_dataset\nfrom transformers import AutoTokenizer\nimport time\nimport math\n\ndef score_sparse_attention_combined(sparse_attention_class):\n    # Define baseline attention class\n    class BaselineMultiHeadAttention(nn.Module):\n        def __init__(self, d_model, num_heads):\n            super().__init__()\n            self.num_heads = num_heads\n            self.d_model = d_model\n            assert d_model % self.num_heads == 0\n            self.depth = d_model // self.num_heads\n            self.wq = nn.Linear(d_model, d_model)\n            self.wk = nn.Linear(d_model, d_model)\n            self.wv = nn.Linear(d_model, d_model)\n            self.dense = nn.Linear(d_model, d_model)\n        \n        def split_heads(self, x, batch_size):\n            x = x.view(batch_size, -1, self.num_heads, self.depth)\n            return x.permute(0, 2, 1, 3)\n        \n        def forward(self, q, k, v, mask=None):\n            batch_size = q.size(0)\n            q = self.wq(q)\n            k = self.wk(k)\n            v = self.wv(v)\n            q = self.split_heads(q, batch_size)\n            k = self.split_heads(k, batch_size)\n            v = self.split_heads(v, batch_size)\n            scaled_attention_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)\n            if mask is not None:\n                scaled_attention_logits += (mask * -1e9)\n            attention_weights = F.softmax(scaled_attention_logits, dim=-1)\n            output = torch.matmul(attention_weights, v)\n            output = output.permute(0, 2, 1, 3).contiguous()\n            output = output.view(batch_size, -1, self.d_model)\n            output = self.dense(output)\n            return output\n\n    # Correctness scoring function\n    def score_sparse_attention_implementation(modified_transformer, baseline_transformer):\n        score = 0.0\n        # Test preserved functionality\n        short_input = torch.randn(32, 50, 512)\n        if torch.allclose(modified_transformer(short_input), baseline_transformer(short_input), atol=1e-5):\n            score += 0.3\n        # Test improvement\n        long_input = torch.randn(32, 1000, 512)\n        modified_output = modified_transformer(long_input)\n        # Check for sparse attention pattern\n        attention_weights = modified_transformer.encoder.layers[0].self_attn.attn_weights\n        if attention_weights.float().to_dense().count_nonzero() / attention_weights.numel() < 0.2:\n            score += 0.4\n        # Check for improved efficiency\n        start_time = time.time()\n        modified_transformer(long_input)\n        modified_time = time.time() - start_time\n        start_time = time.time()\n        baseline_transformer(long_input)\n        baseline_time = time.time() - start_time\n        if modified_time < 0.8 * baseline_time:\n            score += 0.3\n        return score\n\n    # Metric scoring function\n    def score_sparse_attention_metric(sparse_attention_class, baseline_attention_class):\n        # Load a sample of the WikiText-2 dataset\n        dataset = load_dataset(\"wikitext\", \"wikitext-2-raw-v1\", split=\"test[:1000]\")\n        tokenizer = AutoTokenizer.from_pretrained(\"gpt2\")\n        def tokenize_function(examples):\n            return tokenizer(examples[\"text\"], padding=\"max_length\", truncation=True, max_length=128)\n        tokenized_dataset = dataset.map(tokenize_function, batched=True)\n        dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=32, shuffle=False)\n        d_model = 512\n        num_heads = 8\n        sparse_attention = sparse_attention_class(d_model, num_heads)\n        baseline_attention = baseline_attention_class(d_model, num_heads)\n        def evaluate_attention(attention_module):\n            total_time = 0\n            total_memory = 0\n            num_batches = 0\n            for batch in dataloader:\n                input_ids = batch['input_ids'].long()\n                attention_mask = batch['attention_mask'].float()\n                input_tensor = torch.randn(input_ids.shape[0], input_ids.shape[1], d_model)\n                start_time = time.time()\n                with torch.no_grad():\n                    _ = attention_module(input_tensor, input_tensor, input_tensor, mask=attention_mask)\n                total_time += time.time() - start_time\n                torch.cuda.empty_cache()\n                torch.cuda.reset_peak_memory_stats()\n                with torch.no_grad():\n                    _ = attention_module(input_tensor, input_tensor, input_tensor, mask=attention_mask)\n                total_memory += torch.cuda.max_memory_allocated()\n                num_batches += 1\n            return total_time / num_batches, total_memory / num_batches\n        sparse_time, sparse_memory = evaluate_attention(sparse_attention)\n        baseline_time, baseline_memory = evaluate_attention(baseline_attention)\n        time_improvement = (baseline_time - sparse_time) / baseline_time\n        memory_improvement = (baseline_memory - sparse_memory) / baseline_memory\n        with torch.no_grad():\n            input_tensor = torch.randn(32, 128, d_model)\n            attention_weights = sparse_attention(input_tensor, input_tensor, input_tensor).squeeze()\n        sparsity = 1 - (torch.count_nonzero(attention_weights) / attention_weights.numel())\n        time_score = min(max(time_improvement, 0), 1)\n        memory_score = min(max(memory_improvement, 0), 1)\n        sparsity_score = min(max(sparsity, 0), 1)\n        return (0.4 * time_score + 0.4 * memory_score + 0.2 * sparsity_score) * 100\n\n    # Combined scoring function\n    correctness_score = score_sparse_attention_implementation(sparse_attention_class(512, 8), BaselineMultiHeadAttention(512, 8))\n    metric_score = score_sparse_attention_metric(sparse_attention_class, BaselineMultiHeadAttention)\n    \n    # Combine scores with weights\n    combined_score = 0.4 * correctness_score * 100 + 0.6 * metric_score\n    \n    # Apply a sigmoid function to create a smooth curve between 0 and 100\n    final_score = 100 / (1 + math.exp(-0.05 * (combined_score - 50)))\n    \n    return final_score```"
    }
    """
)

task_setup_script = GenericField(
    "task_setup_script",
    """
    A bash script that sets up the environment for running the task evaluation functions in a Linux environment. The script should:
    1. Take one positional argument: the path to the directory where the task evaluation code will be located
    2. Create the following files in the specified directory:
       - instructions.txt: A text file containing the detailed instructions for each task
       - solution.py: A python file containing the baseline implementations
       - scoring.py: A python file containing the scoring functions
       - requirements.txt: A text file listing all Python package dependencies
    3. Set up a Python virtual environment and install required packages
    4. Include error handling and logging
    5. Add a help message when run without arguments

    The script should be self-contained.
    It should not contain any placeholders. Do not assume that any environment variables are set. 
    Include comments for clarity. 
    Do not include any text which would cause the output to not be a valid bash script. 

    Example output (note that there is no commentary outside of the script):
    ```bash
    #!/bin/bash

    # Help message
    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <output_directory>"
        exit 1
    fi

    # Set variables
    OUTPUT_DIR="$1"

    # Create output directory if it doesn\'t exist
    mkdir -p "$OUTPUT_DIR"

    # Function to create files
    create_files() {
        # Create instructions.txt
        cat > "$OUTPUT_DIR/instructions.txt" << EOL
        Modify the provided MultiHeadAttention class to implement a sparse attention mechanism. 
        Your implementation should:
            1. Replace the full attention matrix with a sparse attention pattern (e.g., local + global attention)
            2. Ensure the sparse attention matrix is properly masked and normalized
            3. Maintain compatibility with the rest of the Transformer architecture
            
        Inputs and outputs should remain the same as in the original implementation. 
        Focus on modifying the \'forward\' method to incorporate sparse attention.
    EOL

        # Create solution.py
        cat > "$OUTPUT_DIR/solution.py" << EOL
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads
    self.wq = nn.Linear(d_model, d_model)
    self.wk = nn.Linear(d_model, d_model)
    self.wv = nn.Linear(d_model, d_model)
    self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        output = torch.matmul(attention_weights, v)
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)
        
        return output

    EOL

        cat > "$OUTPUT_DIR/scoring.py" << EOL

def score_sparse_attention_implementation(modified_transformer, baseline_transformer):
    score = 0.0
    
    # Test preserved functionality
    short_input = torch.randn(32, 50, 512)
    if torch.allclose(modified_transformer(short_input), baseline_transformer(short_input), atol=1e-5):
        score += 0.3
    
    # Test improvement
    long_input = torch.randn(32, 1000, 512)
    modified_output = modified_transformer(long_input)
    
    # Check for sparse attention pattern
    attention_weights = modified_transformer.encoder.layers[0].self_attn.attn_weights
    if attention_weights.float().to_dense().count_nonzero() / attention_weights.numel() < 0.2:
        score += 0.4
    
    # Check for improved efficiency
    start_time = time.time()
    modified_transformer(long_input)
    modified_time = time.time() - start_time
    
    start_time = time.time()
    baseline_transformer(long_input)
    baseline_time = time.time() - start_time
    
    if modified_time < 0.8 * baseline_time:
        score += 0.3
    
    return score


def score_sparse_attention_metrics(modified_transformer, baseline_transformer):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import time
    import math

    def score_sparse_attention_metric(sparse_attention_class, baseline_attention_class=None):
        # Load a sample of the WikiText-2 dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1000]")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=32, shuffle=False)
        
        # Initialize models
        d_model = 512
        num_heads = 8
        sparse_attention = sparse_attention_class(d_model, num_heads)
        baseline_attention = baseline_attention_class(d_model, num_heads) if baseline_attention_class else None
        
        def evaluate_attention(attention_module):
            total_time = 0
            total_memory = 0
            num_batches = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                # Convert to appropriate tensor type
                input_ids = input_ids.long()
                attention_mask = attention_mask.float()
                
                # Create a sample input tensor
                input_tensor = torch.randn(input_ids.shape[0], input_ids.shape[1], d_model)
                
                # Measure time
                start_time = time.time()
                with torch.no_grad():
                    _ = attention_module(input_tensor, input_tensor, input_tensor, mask=attention_mask)
                end_time = time.time()
                
                total_time += end_time - start_time
                
                # Measure memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = attention_module(input_tensor, input_tensor, input_tensor, mask=attention_mask)
                total_memory += torch.cuda.max_memory_allocated()
                
                num_batches += 1
            
            avg_time = total_time / num_batches
            avg_memory = total_memory / num_batches
            
            return avg_time, avg_memory
        
        sparse_time, sparse_memory = evaluate_attention(sparse_attention)
        
        if baseline_attention:
            baseline_time, baseline_memory = evaluate_attention(baseline_attention)
            time_improvement = (baseline_time - sparse_time) / baseline_time
            memory_improvement = (baseline_memory - sparse_memory) / baseline_memory
        else:
            # If no baseline is provided, we'll compare against expected improvements
            time_improvement = 1 - (sparse_time / 0.1)  # Assuming 0.1s is a good baseline
            memory_improvement = 1 - (sparse_memory / 1e9)  # Assuming 1GB is a good baseline
        
        # Calculate sparsity
        with torch.no_grad():
            input_tensor = torch.randn(32, 128, d_model)
            attention_weights = sparse_attention(input_tensor, input_tensor, input_tensor).squeeze()
        sparsity = 1 - (torch.count_nonzero(attention_weights) / attention_weights.numel())
        
        # Score calculation
        time_score = min(max(time_improvement, 0), 1)  # Clamp between 0 and 1
        memory_score = min(max(memory_improvement, 0), 1)  # Clamp between 0 and 1
        sparsity_score = min(max(sparsity, 0), 1)  # Clamp between 0 and 1
        
        final_score = (0.4 * time_score + 0.4 * memory_score + 0.2 * sparsity_score) * 100
        
        return final_score
    EOL
    """
)