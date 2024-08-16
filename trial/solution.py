import torch

order = None
def alternating_flip(images, epoch):
    if order is None:
        order = random.permutation(images.size(0))
    return torch.where(order < images.size(0) // 2, images.flip(-1), images) 

# Baseline random flip implementation for comparison
def random_flip(images):
    return torch.where(torch.rand(images.size(0), 1, 1, 1) < 0.5, images.flip(-1), images)
