import torch
import torch.nn as nn
from labml import monit, logger, lab

from labml.logger import Text

from labml_nn.sampling import Sampler
from labml_nn.sampling.temperature import TemperatureSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from threshold_sampler import ThresholdSampler


@torch.no_grad()
def sample(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, sampler: Sampler,
           n_samples: int, n_tokens: int, seq_len: int, prompt: str):
    """
    ## Sample from model

    :param model: is the model to sample from
    :param tokenizer: is the tokenizer to use
    :param sampler: is the sampler to use
    :param n_samples: is the number of samples to generate
    :param n_tokens: is the number of tokens to generate
    :param seq_len: is the maximum sequence length for the model
    :param prompt: is the starting prompt
    """
    # Tokenize the `prompt` and make `n_samples` copies of it
    data = torch.tile(torch.tensor(tokenizer.encode(prompt))[None, :], (n_samples, 1))

    # Collect output for printing
    logs = [[(prompt, Text.meta)] for _ in range(n_samples)]
    # Sample `n_tokens`
    for i in monit.iterate('Sample', n_tokens):
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
        # Decode and add the sampled token for logging
        for j in range(n_samples):
            logs[j] += [('' + tokenizer.decode(res[j]), Text.value)]

    # Print the sampled outputs
    for j in range(n_samples):
        logger.log(logs[j])

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


def main():
    """
    ### Try different sampling techniques
    """

    # Load the model and tokenizer
    with monit.section('Load tokenizer/model'):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
        model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
    # Set the model to eval mode
    model.eval()

    # Prompts to use for sampling
    # prompt = 'I saw an interesting dream last night. '
    prompt = 'It was raining and I heard a loud sound.'

    # Your new Threshold Sampler experimentation
    threshold = 0.009
    target_perplexity = 12.38
    tolerance = 0.01
    base_sampler = TemperatureSampler(1.0)  # or any other base sampler you prefer
    current_perplexity = 0
    with monit.section(f'threshold={threshold}'):
        threshold_sampler = ThresholdSampler(threshold, base_sampler)
        sample(model, tokenizer, threshold_sampler, 4, 32, 128, prompt)
        # Calculate the perplexity of generated text. You need to implement this function.
        current_perplexity = calculate_perplexity(model, tokenizer, threshold_sampler, prompt)
        print(threshold, current_perplexity)

    # while abs(current_perplexity - target_perplexity) > tolerance:
    #     with monit.section(f'threshold={threshold}'):
    #         threshold_sampler = ThresholdSampler(threshold, base_sampler)
    #         sample(model, tokenizer, threshold_sampler, 4, 32, 128, prompt)
    #         # Calculate the perplexity of generated text. You need to implement this function.
    #         current_perplexity = calculate_perplexity(model, tokenizer, threshold_sampler, prompt)
    #         print(threshold, current_perplexity)
    #         if current_perplexity > target_perplexity:
    #             threshold -= 0.01
    #         else:
    #             threshold += 0.01

if __name__ == '__main__':
    main()