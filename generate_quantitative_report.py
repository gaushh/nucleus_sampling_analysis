from collections import defaultdict
import csv
import json
import os

def get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def calculate_repetition_metrics(sentences, repetition_threshold=2, n=1):
    repeated_ngrams_count = 0
    total_ngrams_count = 0

    for sentence in sentences:
        tokens = sentence.split()
        ngrams = get_ngrams(tokens, n)
        ngram_counts = defaultdict(int)

        for ngram in ngrams:
            ngram_counts[ngram] += 1

        repeated_ngrams = [ngram for ngram, count in ngram_counts.items() if count >= repetition_threshold]
        repeated_ngrams_count += len(repeated_ngrams)
        total_ngrams_count += len(ngrams)

    repetition_percentage = (repeated_ngrams_count / total_ngrams_count) * 100

    return repeated_ngrams_count, total_ngrams_count, repetition_percentage


directory = 'conditional_results'
output_csv = 'quantitative_results.csv'

# Iterate over files in the directory
with open(output_csv, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['File', 'n', 'Repeated N-grams', 'Total N-grams', 'Repetition Percentage'])

    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            string_list = []
            with open(filepath, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    string_list.append(data['string'])

            for n in range(1, 6):  # Modify the range as needed
                print(n)
                # Pass the string list to calculate repetition metrics with the current value of n
                repeated_ngrams, total_ngrams, repetition_percentage = calculate_repetition_metrics(string_list, n=n)

                writer.writerow([filename, n, repeated_ngrams, total_ngrams, repetition_percentage])