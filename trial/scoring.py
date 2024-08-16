import torch
import time
from numpy import random


order = None
def alternating_flip(images, epoch):
    if order is None:
        order = random.permutation(images.size(0))
    return torch.where(order < images.size(0) // 2, images.flip(-1), images) 

# Baseline random flip implementation for comparison
def random_flip(images):
    return torch.where(torch.rand(images.size(0), 1, 1, 1) < 0.5, images.flip(-1), images)

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

