from collections import defaultdict
from operator import itemgetter
import json
import os

def get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def rank_sentences(sentences, top_n=100, repetition_threshold=2, n=1):
    sentence_scores = []

    for sentence in sentences:
        tokens = sentence.split()
        token_counts = defaultdict(int)

        for token in tokens:
            token_counts[token] += 1

        ngrams = get_ngrams(tokens, n)
        repeated_ngrams = [ngram for ngram, count in token_counts.items() if count >= repetition_threshold]
        repetition_score = len(repeated_ngrams) / len(ngrams) if len(ngrams) > 0 else 0

        sentence_scores.append((sentence, repetition_score))

    ranked_sentences = sorted(sentence_scores, key=itemgetter(1), reverse=True)[:top_n]
    ranked_sentences = [sentence for sentence, _ in ranked_sentences]

    return ranked_sentences


directory = 'conditional_results'
output_directory = 'nucleus_report'

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jsonl'):
        filepath = os.path.join(directory, filename)
        string_list = []
        with open(filepath, 'r') as file:
            for line in file:
                data = json.loads(line)
                string_list.append(data['string'])

        for n in range(1, 6):  # Modify the range as needed
            print("GRAM:: ", n)
            # Create subdirectory for each value of n
            n_directory = os.path.join(output_directory, f'n_{n}')
            os.makedirs(n_directory, exist_ok=True)

            # Pass the string list to the rank_sentences function with the current value of n
            result_list = rank_sentences(string_list, n=n)

            output_file = os.path.join(n_directory, f'ranked_{filename[:-6]}.txt')
            # Save the result list to a text file in the subdirectory
            with open(output_file, 'w') as file:
                for item in result_list:
                    file.write("%s\n" % item)
