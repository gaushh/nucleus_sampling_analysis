from labml import monit, logger, lab
from labml_nn.sampling.temperature import TemperatureSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from threshold_sampler import ThresholdSampler
import matplotlib.pyplot as plt
import random
from typing import List
from utils1 import read_json_file, calculate_perplexity, sample
import tqdm
NEWLINE = 198

def sublist_end_index(list1, list2):
    s1, s2 = ' '.join(map(str, list1)), ' '.join(map(str, list2))
    if s1 in s2:
        return s2[:s2.index(s1)].count(' ') + s1.count(' ') + 1
    else:
        return None


def moving_threshold(dataset: List[str], num_samples: int = 10):
    # Load the model and tokenizer
    with monit.section('Load tokenizer/model'):
        print("Loading tokenizer and model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
        model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
        print("Done loading tokenizer and model...")
    # Set the model to eval mode
    model.eval()
    # Your new Threshold Sampler experimentation
    threshold = 0.09
    target_perplexity = 12.38
    tolerance = 0.01
    base_sampler = TemperatureSampler(1.0)  # or any other base sampler you prefer
    current_perplexity = 0
    perplexities = []
    thresholds = []
    min_diff = float('inf')
    optimal_threshold = threshold
    # Sample num_samples random rows from the dataset
    sampled_rows = random.sample(dataset, num_samples)
    while threshold > tolerance:
        avg_perplexity = 0
        with monit.section(f'threshold={threshold}'):
            threshold_sampler = ThresholdSampler(threshold, base_sampler)
            for row in sampled_rows:

                row = row[:40]  # Ensure prompt length is not more than 40
                sample(model, tokenizer, threshold_sampler, 1, 100, 40, row)
                current_perplexity = calculate_perplexity(model, tokenizer, threshold_sampler, row, 1, 200, 40)
                print(f"Current perplexity: {current_perplexity}")
                avg_perplexity += current_perplexity

            avg_perplexity /= num_samples
            print(f"Average perplexity: {avg_perplexity}")
            perplexities.append(avg_perplexity)
            thresholds.append(threshold)

            if abs(avg_perplexity - target_perplexity) < min_diff:
                min_diff = abs(avg_perplexity - target_perplexity)
                optimal_threshold = threshold

            threshold -= 0.01

    # Plotting
    plt.plot(thresholds, perplexities)
    plt.xlabel('Threshold')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. Threshold')
    plt.show()

    print(f"Optimal threshold: {optimal_threshold}")


def static_threshold(dataset: List[str]):
    with monit.section('Load tokenizer/model'):
        print("Loading tokenizer and model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
        model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
        print("Done loading tokenizer and model...")
    model.eval()
    threshold = 0.09
    base_sampler = TemperatureSampler(1.0)
    perplexities = []
    min_diff = float('inf')
    optimal_threshold = threshold
    rows = dataset
    # print("length of rows ", len(rows))
    avg_perplexity = 0
    threshold_sampler = ThresholdSampler(threshold, base_sampler)
    for row in tqdm.tqdm(rows[:100]):
        # print(row,"//")
        # row = row[:40]  # Ensure prompt length is not more than 40
        current_perplexity = calculate_perplexity(model, tokenizer, threshold_sampler, row, 1, 200, 40)
        perplexities.append(current_perplexity)
        avg_perplexity += current_perplexity

    avg_perplexity /= len(rows)
    print(f"Average perplexity: {avg_perplexity}")
    # Plotting
    plt.plot(len(rows), perplexities)
    plt.xlabel('Data Samples')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. Documents')
    plt.show()


if __name__ == '__main__':
    # Assume your dataset is a list of strings
    decoded_dataset = read_json_file("tokenized_dataset.json")
    static_threshold(decoded_dataset)


