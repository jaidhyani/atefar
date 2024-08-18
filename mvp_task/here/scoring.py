from regex import F
import torch
import time
import timeit

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
            print("Mean preservation detected")
            score += 0.2
        
        # Check alternating behavior
        even_epoch_flips = implementation(images, epoch=2)
        odd_epoch_flips = implementation(images, epoch=3)
        if not torch.allclose(even_epoch_flips, odd_epoch_flips):
            print("Alternating flip pattern detected")
            score += 0.3
        
        # Check consistency across epochs
        if torch.allclose(implementation(images, epoch=2), implementation(images, epoch=4)):
            print("Consistent flip behavior detected")
            score += 0.2
        
        # Check performance
        baseline_time = timeit.timeit(lambda: baseline(images), number=100)
        implementation_time = timeit.timeit(lambda: implementation(images, epoch=1), number=100)
        if implementation_time < baseline_time:
            print("Speed improvement detected")
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
            improvement_bonus = 50 * (baseline_time - implementation_time) / baseline_time
            score += improvement_bonus
            print(f"Speed improvement bonus: {improvement_bonus}")
        
        # Check correctness
        flipped_even = implementation(images, 2)
        flipped_odd = implementation(images, 3)
        if torch.sum(flipped_even == flipped_odd) / flipped_even.numel() < 0.1:
            print("Alternating flip pattern detected")
            score += 25
        if torch.abs(torch.sum(flipped_even == images) / images.numel() - 0.5) < 0.05:
            print("Mean preservation detected")
            score += 25
        
        return score

    correctness_score = score_correctness(alternating_flip_fn, baseline_random_flip)
    metric_score = score_metric(alternating_flip_fn, baseline_random_flip)
    
    # Combine scores with weights
    combined_score = 0.4 * correctness_score + 0.6 * metric_score
    return combined_score


def random_flip(images, epoch):
    # Standard random flip implementation
    flip_mask = (torch.rand(len(images)) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, images.flip(-1), images)

def jai_flip(images, epoch):
    flip_mask = (
        torch.rand(len(images), generator=torch.Generator().manual_seed(42)) < 0.5
    ).view(-1, 1, 1, 1)
    # flip_mask = (
    #     torch.rand(len(images)) < 0.5
    # ).view(-1, 1, 1, 1)
    if epoch % 2 == 0:
        return torch.where(flip_mask, images.flip(-1), images)
    else:
        return torch.where(flip_mask, images, images.flip(-1))
    return torch.where(flip_mask, images.flip(-1), images)


def haiku_flip(images, epoch):
    """
    Applies the alternating flip augmentation to a batch of images.

    Args:
        images (torch.Tensor): A batch of images, shape (batch_size, channels, height, width).
        epoch (int): The current epoch number.

    Returns:
        torch.Tensor: The batch of augmented images.
    """
    batch_size = images.size(0)

    # For the first epoch, randomly flip 50% of the images
    if epoch == 0:
        flip_mask = (torch.rand(batch_size) < 0.5).view(-1, 1, 1, 1)
        return torch.where(flip_mask, images.flip(-1), images)

    # For subsequent epochs, deterministically flip based on the first epoch
    else:
        # Use a pseudorandom function based on image indices to determine which images to flip
        flip_indices = torch.arange(batch_size).to(images.device)
        flip_mask = (flip_indices % 2 == (epoch % 2)).view(-1, 1, 1, 1)
        return torch.where(flip_mask, images.flip(-1), images)

def chatgptmini_flip(images, epoch):
    batch_size = len(images)
    
    # Determine the flip mask based on the epoch
    if epoch == 0:
        # First epoch: Randomly flip 50% of the images
        flip_mask = (torch.rand(batch_size) < 0.5)
        # Store flip mask for use in future epochs
        # This would be done in practice by saving the mask somewhere, for now we assume it's available
    else:
        # For subsequent epochs: Flip based on previous flip mask
        # For deterministic results, use a fixed pseudorandom function based on the epoch and image indices
        flip_mask = (torch.arange(batch_size) % 2 == (epoch % 2))
    
    # Apply the flip based on the determined mask
    flip_mask = flip_mask.view(-1, 1, 1, 1)  # Adjust dimensions for broadcasting
    flipped_images = torch.where(flip_mask, images.flip(-1), images)
    
    return flipped_images

print("baseline")
print(score_alternating_flip_combined(random_flip))
print("jai_flip")
print(score_alternating_flip_combined(jai_flip))
print("haiku_flip")
print(score_alternating_flip_combined(haiku_flip))
print("chatgptmini_flip")
print(score_alternating_flip_combined(chatgptmini_flip))