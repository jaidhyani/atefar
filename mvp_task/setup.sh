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
Implement the alternating flip augmentation method as described in the paper. Your implementation should:
1. Take a batch of images and the current epoch number as inputs.
2. For the first epoch, randomly flip 50% of the images horizontally.
3. For subsequent epochs, deterministically flip images based on whether they were flipped in the first epoch:
   - On even epochs, flip only those images that were not flipped in the first epoch.
   - On odd epochs, flip only those images that were flipped in the first epoch.
4. Use a pseudorandom function based on image indices to determine which images to flip, avoiding the need for extra memory.
Your implementation should be efficient and work with PyTorch tensors. Compare your results with the provided baseline random flip implementation to ensure correctness and measure performance improvements. Pay special attention to handling edge cases and ensuring that the alternating pattern is maintained across epochs.
EOL

    # Create solution.py
    cat > "$OUTPUT_DIR/solution.py" << EOL
import torch

def random_flip(images):
    # Standard random flip implementation
    flip_mask = (torch.rand(len(images)) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, images.flip(-1), images)

def alternating_flip(images, epoch):
    # TODO: Implement alternating flip logic here
    # For now, this just calls random_flip
    return random_flip(images)

# Example usage
batch_size, channels, height, width = 32, 3, 32, 32
images = torch.randn(batch_size, channels, height, width)
epoch = 1
augmented_images = alternating_flip(images, epoch)
EOL

    # Create scoring.py
    cat > "$OUTPUT_DIR/scoring.py" << EOL
import torch
import time

def score_alternating_flip_combined(alternating_flip_fn):
    def baseline_random_flip(images):
        flip_mask = (torch.rand(len(images)) < 0.5).view(-1, 1, 1, 1)
        return torch.where(flip_mask, images.flip(-1), images)

    # Correctness scoring
    def score_correctness(implementation, baseline):
        score = 0.0
        images = torch.randn(1000, 3, 32, 32)
        
        # Check first epoch behavior
        first_epoch_flips = implementation(images, epoch=0)
        if torch.isclose(first_epoch_flips.float().mean(), torch.tensor(0.5), atol=0.05):
            score += 0.2
        
        # Check alternating behavior
        even_epoch_flips = implementation(images, epoch=2)
        odd_epoch_flips = implementation(images, epoch=3)
        if not torch.allclose(even_epoch_flips, odd_epoch_flips):
            score += 0.3
        
        # Check consistency across epochs
        if torch.allclose(implementation(images, epoch=2), implementation(images, epoch=4)):
            score += 0.2
        
        # Check performance
        baseline_time = timeit.timeit(lambda: baseline(images), number=100)
        implementation_time = timeit.timeit(lambda: implementation(images, epoch=1), number=100)
        if implementation_time < baseline_time:
            score += 0.3
        
        return score * 100  # Scale to 0-100

    # Metric scoring
    def score_metric(implementation, baseline):
        score = 0.0
        batch_size, channels, height, width = 1024, 3, 32, 32
        images = torch.randn(batch_size, channels, height, width)
        epochs = 10
        
        # Measure speed
        start_time = time.time()
        for epoch in range(epochs):
            _ = implementation(images, epoch)
        implementation_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(epochs):
            _ = baseline(images)
        baseline_time = time.time() - start_time
        
        if implementation_time < baseline_time:
            score += 50 * (baseline_time - implementation_time) / baseline_time
        
        # Check correctness
        flipped_even = implementation(images, 2)
        flipped_odd = implementation(images, 3)
        if torch.sum(flipped_even == flipped_odd) / flipped_even.numel() < 0.1:
            score += 25
        if torch.abs(torch.sum(flipped_even == images) / images.numel() - 0.5) < 0.05:
            score += 25
        
        return score

    correctness_score = score_correctness(alternating_flip_fn, baseline_random_flip)
    metric_score = score_metric(alternating_flip_fn, baseline_random_flip)
    
    # Combine scores with weights
    combined_score = 0.4 * correctness_score + 0.6 * metric_score
    return combined_score
EOL

    # Create requirements.txt
    cat > "$OUTPUT_DIR/requirements.txt" << EOL
torch
EOL
}

# Create the files
create_files

# Set up Python virtual environment
python3 -m venv "$OUTPUT_DIR/venv"
source "$OUTPUT_DIR/venv/bin/activate"

# Install required packages
pip install -r "$OUTPUT_DIR/requirements.txt"

echo "Task setup complete. Files created in $OUTPUT_DIR"
echo "Activate the virtual environment with: source $OUTPUT_DIR/venv/bin/activate"
