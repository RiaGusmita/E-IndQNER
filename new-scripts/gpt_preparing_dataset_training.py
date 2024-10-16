import pandas as pd
from datasets import Dataset
import json

# Function to prepare each example
def prepare_examples(example):
    system_message = """
    Given the following entity classes and sentences, label entity mentions with their respective classes in sentences according to the sentences' context. 
    In the output, only include entity mentions and their respective class in the given output format. No needed further explanation.
    CONTEXT: entity classes: {text["named_entity_classes"]}. 
    Example sentence: Jika kamu (tetap) dalam keraguan tentang apa (Al-Qur’an) yang Kami turunkan kepada hamba Kami (Nabi Muhammad), buatlah satu surah yang semisal dengannya dan ajaklah penolong-penolongmu selain Allah, jika kamu orang-orang yang benar.
    Example output: Jika/O kamu/O (/O tetap/O )/O dalam/O keraguan/O tentang/O apa/O (/O Al-Qur’an/HolyBook )/O yang/O Kami/O turunkan/O kepada/O hamba/O Kami/O (/O Nabi/O Muhammad/Messenger )/O ,/O buatlah/O satu/O surah/O yang/O semisal/O dengannya/O dan/O ajaklah/O penolong-penolongmu/O selain/O Allah/Allah ,/O jika/O kamu/O orang-orang/O yang/O benar/O ./O
    """
    
    test_sentence =f"""
    Test sentence: {example['sentence']}
    Test output:
    """
    test_output = example["label"]
    return {
        "messages": [
            {"content": system_message, "role": "system"},
            {"content": test_sentence, "role": "user"},
            {"content": test_output, "role": "assistant"}
        ]
    }

domain = "specific-domain"

# Read CSV files
sentences_df = pd.read_csv("../new-datasets/dev.txt", sep="\t", header=None, names=["sentence"])
labels_df = pd.read_csv("../new-datasets/dev_with_labels.txt", sep="\t", header=None, names=["label"])
df = pd.concat([sentences_df, labels_df], axis=1)

# Convert DataFrame to Dataset
dataset_df = Dataset.from_pandas(df)

# Apply the prepare_examples function
dataset = dataset_df.map(prepare_examples, remove_columns=['sentence', 'label'])

# Extract only the messages field
messages_list = [example for example in dataset]

# Save the messages to a JSONL file
json_output = f"../new-datasets/gpt_dev.jsonl"
with open(json_output, "w") as json_file:
    for message in messages_list:
        json.dump(message, json_file)
        json_file.write("\n")

print(f"Messages saved to {json_output}")
