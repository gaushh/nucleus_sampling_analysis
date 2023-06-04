from labml import monit, logger, lab
from labml_nn.sampling.temperature import TemperatureSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from threshold_sampler import ThresholdSampler
import matplotlib.pyplot as plt
import random
from typing import List
from utils1 import read_json_file, calculate_perplexity, sample
import tqdm
import json
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
    # threshold_values = np.logspace(-3, -2, 10)  # generates 10 values between 10^-3 and 10^-2
    threshold_values = [0.009]  # generates 10 values between 10^-3 and 10^-2
    output_data = []  # create a list to store output text and perplexity for each prompt and threshold

    with monit.section('Load tokenizer/model'):
        print("Loading tokenizer and model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
        model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
        print("Done loading tokenizer and model...")
    model.eval()

    for threshold in tqdm.tqdm(threshold_values):
        base_sampler = TemperatureSampler(1.0)
        threshold_sampler = ThresholdSampler(threshold, base_sampler)
        avg_perplexity = 0
        rows = dataset
        perplexities = []
        avg_perplexities = []


        for row in rows[:100]:
            current_perplexity, output_text = calculate_perplexity(model, tokenizer, threshold_sampler, row, 1, 200, 40)
            perplexities.append(current_perplexity)
            avg_perplexity += current_perplexity
            output_data.append(
                {"prompt": row, "output_text": output_text, "perplexity": current_perplexity, "threshold": threshold})

        avg_perplexity /= len(rows)
        avg_perplexities.append(avg_perplexity)


    # Save output_data into a JSON file
    with open("output_data.json", "w") as file:
        json.dump(output_data, file, indent=4)

    plt.figure(figsize=(10, 5))
    plt.plot(threshold_values, avg_perplexities, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Average Perplexity')
    plt.title('Average Perplexity vs Threshold')
    plt.grid(True)
    plt.savefig("perplexity_vs_threshold.png")
    plt.show()


if __name__ == '__main__':
    # Assume your dataset is a list of strings
    decoded_dataset = read_json_file("tokenized_dataset.json")
    static_threshold(decoded_dataset)


