import torch
import numpy as np



def random_flip(images):
    # Standard random flip implementation
    flip_mask = (torch.rand(len(images)) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, images.flip(-1), images)

def alternating_flip(images, epoch):
    flip_mask = (
        torch.rand(len(images), generator=torch.Generator().manual_seed(42)) < 0.5
    ).view(-1, 1, 1, 1)
    if epoch % 2:
        return torch.where(flip_mask, images.flip(-1), images)
    else:
        return torch.where(flip_mask, images, images.flip(-1))

# Example usage
batch_size, channels, height, width = 32, 3, 32, 32
images = torch.randn(batch_size, channels, height, width)
epoch = 1
augmented_images = alternating_flip(images, epoch)
