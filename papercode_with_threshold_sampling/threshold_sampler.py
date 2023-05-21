import torch
from torch import nn

from labml_nn.sampling import Sampler

class ThresholdSampler(Sampler):
    """
    ## Threshold Sampler
    """
    def __init__(self, threshold: float, sampler: Sampler):
        """
        :param threshold: is the probability threshold to select tokens
        :param sampler: is the sampler to use for the selected tokens
        """
        self.threshold = threshold
        self.sampler = sampler
        # Softmax to compute $P(x_i | x_{1:i-1})$ from the logits
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits with Threshold Sampling
        """

        # Get probabilities $P(x_i | x_{1:i-1})$
        probs = self.softmax(logits)

        # Create a mask where probabilities are above the threshold
        threshold_mask = probs > self.threshold

        # Apply the mask to the logits, setting those below the threshold to a very low value instead of '-inf'
        masked_logits = logits.clone()
        masked_logits[~threshold_mask] = float('-1e20')  # or any very low value of your preference

        # Sample from the sampler
        res = self.sampler(masked_logits)

        return res
