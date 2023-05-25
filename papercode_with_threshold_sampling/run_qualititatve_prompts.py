import torch

from labml import monit, logger, lab

from labml.logger import Text

from labml_nn.sampling import Sampler
from labml_nn.sampling.greedy import GreedySampler
from labml_nn.sampling.nucleus import NucleusSampler
from labml_nn.sampling.temperature import TemperatureSampler
from labml_nn.sampling.top_k import TopKSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import calculate_perplexity
from threshold_sampler import ThresholdSampler

@torch.no_grad()
def sample(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, sampler: Sampler,
           n_samples: int, n_tokens: int, seq_len: int, prompt: str, method: str):
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
    outs = []
    for i in range(n_samples):
        outs.append(prompt+' ')
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
            outs[j] += '' + tokenizer.decode(res[j])

    # Print the sampled outputs
    with open("Output.txt", "a") as text_file:
        text_file.write(method+'\n')
        for out in outs:
            text_file.write(out+'\n')
    for j in range(n_samples):
        logger.log(logs[j])


def main():
    """
    ### Run sampling different sampling techniques and save outputs to a file
    """

    # Load the model and tokenizer
    with monit.section('Load tokenizer/model'):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
        model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
    # Set the model to eval mode
    model.eval()

    # Prompts to use for sampling
    # prompt = 'I saw an interesting dream last night. '
    prompts = ['I saw an interesting dream last night. ', 'It was a long time ago, ', 'It was a rainy day and I heard a loud sound.','The food was delicious. ', 'We were all tired. ']
    for prompt in prompts:

        # [Greedy Sampling](greedy.html)
        with monit.section('greedy'):
            sample(model, tokenizer, GreedySampler(), 4, 32, 128, prompt, method='greedy')

        # [Temperature Sampling](temperature.html)
        with monit.section('temperature=1.'):
            sample(model, tokenizer, TemperatureSampler(1.), 4, 32, 128, prompt, method = 'temperature=1.')
        with monit.section('temperature=.1'):
            sample(model, tokenizer, TemperatureSampler(.1), 4, 32, 128, prompt, method = 'temperature=.1')
        with monit.section('temperature=10.'):
            sample(model, tokenizer, TemperatureSampler(10.), 4, 32, 128, prompt, method = 'temperature=10.')

        # [Top-k Sampling](top_k.html)
        with monit.section('top_k=5'):
            sample(model, tokenizer, TopKSampler(2, TemperatureSampler(1.)), 4, 32, 128, prompt, method= 'top_k=5')

        # [Nucleus Sampling](nucleus.html)
        with monit.section('nucleus p=.95'):
            sample(model, tokenizer, NucleusSampler(0.95, TemperatureSampler(1.)), 4, 32, 128, prompt, method = 'nucleus_p=0.95')
        with monit.section('nucleus p=.1'):
            sample(model, tokenizer, NucleusSampler(0.1, TemperatureSampler(1.)), 4, 32, 128, prompt, method = 'nucleus_p=.1')

        # Thresholding technique
        threshold = 0.01
        with monit.section(f'threshold={threshold}'):
            threshold_sampler = ThresholdSampler(threshold, TemperatureSampler(1.))
            sample(model, tokenizer, threshold_sampler, 4, 32, 128, prompt, method='threshold=.01')
        
    

#
if __name__ == '__main__':
    main()
