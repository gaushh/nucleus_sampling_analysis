import json
import random
from typing import List

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils1 import read_jsonl

NEWLINE = [198, 198]  # tokenized form of "\n\n"


def sublist_end_index(list1, list2):
    s1, s2 = ' '.join(map(str, list1)), ' '.join(map(str, list2))
    if s1 in s2:
        return s2[:s2.index(s1)].count(' ') + s1.count(' ') + 1
    else:
        return None


# def main(dataset: List[str], num_samples: int = 5000):
#     # Load the tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
#
#     sampled_rows = random.sample(dataset, num_samples)
#     # sampled_rows = dataset
#
#     tokenized_dataset = []
#     decoded_dataset = []
#
#     for row in sampled_rows:
#         # Tokenize each sample
#         tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row))
#         idx = sublist_end_index(NEWLINE, tokens)
#         # Truncate to index of "\n\n" if it exists within 40 tokens, else truncate to 40 tokens
#         if idx is not None and idx < 40:
#             tokens = tokens[:idx]
#         else:
#             tokens = tokens[:40]
#
#         tokenized_dataset.append(tokens)
#         # Decode the tokenized sample back to string
#         decoded_dataset.append(tokenizer.decode(tokens))
#
#     # Save the tokenized dataset to a file
#     with open("tokenized_dataset.json", 'w') as file:
#         json.dump(tokenized_dataset, file)
#
#     # Save the decoded dataset to another file
#     with open("decoded_dataset.json", 'w') as file:
#         json.dump(decoded_dataset, file)

def main(dataset: List[str], num_samples: int = 5000):
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)

    sampled_rows = dataset
    # sampled_rows = dataset

    tokenized_dataset = []
    decoded_dataset = []

    for row in sampled_rows:
        # tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row))
        # idx = sublist_end_index(NEWLINE, tokens)
        # # Truncate to index of "\n\n" if it exists within 40 tokens, else truncate to 40 tokens
        # if idx is not None and idx < 40:
        #     tokens = tokens[:idx]
        # else:
        #     tokens = tokens[:40]

        tokenized_dataset.append(row)
        # Decode the tokenized sample back to string
        decoded_dataset.append(tokenizer.decode(row))

    # Save the tokenized dataset to a file
    with open("tokenized_dataset.json", 'w') as file:
        json.dump(tokenized_dataset, file)

    # Save the decoded dataset to another file
    with open("decoded_dataset.json", 'w') as file:
        json.dump(decoded_dataset, file)


if __name__ == '__main__':
    # Assume your dataset is a list of strings
    dataset = read_jsonl()
    main(dataset)
