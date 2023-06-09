import jsonlines
from collections import defaultdict
import csv
import json
import os
from collections import defaultdict

def get_ngrams(tokens, n):
    # Implement your n-gram generation logic here
    # Return a list of n-grams based on the given tokens and n
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def calculate_repetitiveness(string, repetition_threshold=2, n=1):
    tokens = string.split()
    ngrams = get_ngrams(tokens, n)
    ngram_counts = defaultdict(int)

    for ngram in ngrams:
        ngram_counts[ngram] += 1

    repeated_ngrams = [ngram for ngram, count in ngram_counts.items() if count >= repetition_threshold]
    repeated_ngrams_count = len(repeated_ngrams)
    total_ngrams_count = len(ngrams)

    if total_ngrams_count == 0:
        repetition_percentage = 0.0
    else:
        repetition_percentage = (repeated_ngrams_count / total_ngrams_count) * 100

    return repetition_percentage




def process_file(input_file, output_dir):
    filename = os.path.splitext(os.path.basename(input_file))[0]

    for n in range(1, 6):
        print(n)
        output_file = os.path.join(output_dir, f"{filename}_ngram_{n}.csv")

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['repetitiveness_score', 'len', 'ppl'])

            with jsonlines.open(input_file) as reader:
                for obj in reader:
                    string = obj['string']
                    repetitiveness_score = calculate_repetitiveness(string, n)
                    row = [repetitiveness_score, obj['len'], obj['ppl']]
                    writer.writerow(row)


def main():
    input_folder = 'conditional_results'
    output_folder = 'results/perplexity_analysis'

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jsonl'):
                input_file = os.path.join(root, file)
                output_dir = os.path.join(output_folder, os.path.splitext(file)[0])

                os.makedirs(output_dir, exist_ok=True)
                process_file(input_file, output_dir)


if __name__ == '__main__':
    main()
