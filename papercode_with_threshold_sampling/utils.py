import torch
import torch.nn as nn
from labml import monit, logger, lab

from labml.logger import Text

from labml_nn.sampling import Sampler
from labml_nn.sampling.temperature import TemperatureSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from threshold_sampler import ThresholdSampler

def calculate_perplexity(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, sampler: Sampler, prompt: str,
                         n_samples: int = 1, n_tokens: int = 100, seq_len: int = 128):
    model.eval()

    # Tokenize the `prompt` and make `n_samples` copies of it
    data = torch.tile(torch.tensor(tokenizer.encode(prompt))[None, :], (n_samples, 1))
    total_log_prob = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(n_tokens):
            # Truncate the data to the maximum sequence length
            data = data[-seq_len:]
            # Get the model output. The 'logits' has shape `[batch_size, seq_len, n_tokens]`
            logits = model(data)[0]
            # Get the `logits` of the last token
            logits = logits[:, -1]
            # Sample from the `logits`
            res = sampler(logits)
            # Add the sampled token to the data
            data = torch.cat([data, res[:, None]], dim=1)

            # Calculate log probabilities
            log_probs = nn.LogSoftmax(dim=-1)(logits)
            # Gather the log probabilities of sampled tokens
            sampled_log_probs = log_probs.gather(-1, res.unsqueeze(-1)).squeeze(-1)
            total_log_prob += sampled_log_probs.sum().item()
            total_tokens += res.shape[0]

    # Calculate average negative log probability
    avg_neg_log_prob = -total_log_prob / total_tokens
    # Perplexity is the exponentiation of the average negative log probability
    perplexity = torch.exp(torch.tensor(avg_neg_log_prob))

    return perplexity.item()