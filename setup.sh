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
Implement the alternating flip augmentation method as described in Section 3.6 of the paper. Your implementation should:
1. Take a batch of images and the current epoch number as input.
2. For the first epoch, randomly flip 50% of the images horizontally.
3. For subsequent epochs, deterministically flip images based on whether they were flipped in the first epoch:
   - On even epochs (2, 4, 6, ...), flip only those images that were not flipped in the first epoch.
   - On odd epochs (3, 5, 7, ...), flip only those images that were flipped in the first epoch.
4. Use a pseudorandom function based on image indices to determine which images to flip, avoiding the need for extra memory.
5. Ensure that your implementation is efficient and can handle large batch sizes.
6. Compare the performance of your implementation with the provided baseline random flip augmentation in terms of both accuracy improvement and computational efficiency.
Your function should be compatible with PyTorch\'s data augmentation pipeline and should be able to be easily integrated into the training loop.
EOL

    # Create solution.py
    cat > "$OUTPUT_DIR/solution.py" << EOL
import torch

def alternating_flip(images, epoch):
    # TODO: Implement alternating flip augmentation here
    pass

# Baseline random flip implementation for comparison
def random_flip(images):
    return torch.where(torch.rand(images.size(0), 1, 1, 1) < 0.5, images.flip(-1), images)
EOL

    # Create scoring.py
    cat > "$OUTPUT_DIR/scoring.py" << EOL
import torch
import time

def score_alternating_flip(implementation, baseline):
    score = 0.0
    # Check flipping pattern
    for epoch in range(10):
        images = torch.randn(1000, 3, 32, 32)
        flipped = implementation(images, epoch)
        if epoch == 0:
            if torch.allclose(flipped.float().mean(), images.float().mean(), atol=1e-2):
                score += 0.2
        elif epoch % 2 == 1:
            if torch.allclose(flipped[:500], images[:500]) and torch.allclose(flipped[500:].flip(-1), images[500:]):
                score += 0.1
        else:
            if torch.allclose(flipped[:500].flip(-1), images[:500]) and torch.allclose(flipped[500:], images[500:]):
                score += 0.1

    # Measure speed improvement
    start_time = time.time()
    for _ in range(100):
        implementation(images, 1)
    impl_time = time.time() - start_time

    start_time = time.time()
    for _ in range(100):
        baseline(images)
    base_time = time.time() - start_time

    if impl_time < base_time:
        score += 0.2

    # Note: Actual model training and evaluation are omitted for simplicity
    # In a real scenario, you would train models with both augmentations and compare accuracies

    return score

def score_alternating_flip_metrics(implementation, baseline, dataset):
    # Simplified metric scoring function
    # In a real scenario, you would train models and measure actual accuracy improvements
    acc_improvement = 0.05  # Placeholder value

    # Measure speed improvement
    start_time = time.time()
    for _ in range(1000):
        implementation(dataset[0], 1)
    impl_time = time.time() - start_time

    start_time = time.time()
    for _ in range(1000):
        baseline(dataset[0])
    base_time = time.time() - start_time

    speed_improvement = (base_time - impl_time) / base_time

    score = 0.6 * acc_improvement + 0.4 * speed_improvement
    return max(0, min(1, score))

def score_alternating_flip_combined(implementation, baseline, dataset):
    correctness_score = score_alternating_flip(implementation, baseline)
    metric_score = score_alternating_flip_metrics(implementation, baseline, dataset)
    return 0.6 * correctness_score + 0.4 * metric_score
EOL

    # Create requirements.txt
    cat > "$OUTPUT_DIR/requirements.txt" << EOL
torch
numpy
EOL
}

# Function to set up Python environment
setup_python_env() {
    python3 -m venv "$OUTPUT_DIR/venv"
    source "$OUTPUT_DIR/venv/bin/activate"
    pip install -r "$OUTPUT_DIR/requirements.txt"
}

# Main execution
main() {
    create_files
    setup_python_env
    echo "Setup complete. Evaluation environment is ready in $OUTPUT_DIR"
}

# Run the script
main