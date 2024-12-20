import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

import warnings
import os
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
import ast
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="resume_download is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="The use_auth_token argument is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="--push_to_hub_token is deprecated")

# Set CUDA_LAUNCH_BLOCKING for debugging
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
base_model_name = "meta-llama/Llama-3.1-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, max_length=512)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Paths to the train dataset files
train_sentences_path = '../new-datasets/train.txt'
train_labels_path = '../new-datasets/updated_train_set_final.txt'

# Read the sentences and labels from train dataset
with open(train_sentences_path, 'r', encoding='utf-8') as f:
    train_sentences = f.readlines()

with open(train_labels_path, 'r', encoding='utf-8') as f:
    train_labels = f.readlines()

# Create DataFrames from the lists of sentences and labels
train_sentences_df = pd.DataFrame(train_sentences, columns=["sentence"])
train_labels_df = pd.DataFrame(train_labels, columns=["label"])

# Combine the DataFrames side by side
train_df = pd.concat([train_sentences_df, train_labels_df], axis=1)


# Paths to the validation dataset files
dev_sentences_path = '../new-datasets/dev.txt'
dev_labels_path = '../new-datasets/updated_val_set_final.txt'

# Read the sentences and labels from test dataset
with open(dev_sentences_path, 'r', encoding='utf-8') as f:
    dev_sentences = f.readlines()

with open(dev_labels_path, 'r', encoding='utf-8') as f:
    dev_labels = f.readlines()

# Create DataFrames from the lists of sentences and labels
dev_sentences_df = pd.DataFrame(dev_sentences, columns=["sentence"])
dev_labels_df = pd.DataFrame(dev_labels, columns=["label"])

# Combine the DataFrames side by side
val_df = pd.concat([dev_sentences_df, dev_labels_df], axis=1)

# Open and read the JSON file
with open("named_entity_class_dictionary.json", "r") as json_file:
    named_entity_classes_dict = json.load(json_file)

# Print the loaded dictionary
named_entity_classes = [named_entity_class for named_entity_class in named_entity_classes_dict]
print("List of entity class")
print(named_entity_classes)

print("Train DataFrame:")
print(train_df.head())

print("\nValidation DataFrame:")
print(val_df.head())

system_message = """
Given the following entity classes and sentences, label entity mentions with their respective classes in sentences according to the sentences' context.
In the output, only include entity mentions and their respective class in the given output format. No needed further explanation.
CONTEXT: entity classes: {named_entity_classes}.
Example sentence: Jika kamu (tetap) dalam keraguan tentang apa (Al-Qur’an) yang Kami turunkan kepada hamba Kami (Nabi Muhammad), buatlah satu surah yang semisal dengannya dan ajaklah penolong-penolongmu selain Allah, jika kamu orang-orang yang benar.
Example output: Jika/O kamu/O (/O tetap/O )/O dalam/O keraguan/O tentang/O apa/O (/O Al-Qur’an/HolyBook )/O yang/O Kami/O turunkan/O kepada/O hamba/O Kami/O (/O Nabi/O Muhammad/Messenger )/O ,/O buatlah/O satu/O surah/O yang/O semisal/O dengannya/O dan/O ajaklah/O penolong-penolongmu/O selain/O Allah/Allah ,/O jika/O kamu/O orang-orang/O yang/O benar/O ./O
"""

# Define the reasoning and entity linking task
def prepare_examples(example):
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": example["sentence"]},
            {"role": "assistant", "content": example["label"]}
        ]
    }

# Convert DataFrame to Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Apply the prepare_examples function
train_dataset = train_dataset.map(prepare_examples, remove_columns=['sentence', 'label'])
val_dataset = val_dataset.map(prepare_examples, remove_columns=['sentence', 'label'])

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_dir = "./results"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
device_map = {"": "cuda"}
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True,
    low_cpu_mem_usage=True
)
base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1

# Use DataParallel to wrap the model
#if torch.cuda.device_count() > 1:
#    base_model = torch.nn.DataParallel(base_model)


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Increased batch size
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=500,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,  # Enable mixed precision training
    bf16=False,
    max_grad_norm=0.3,
    logging_steps=10,
    warmup_ratio=0.03,
    warmup_steps=100,
    group_by_length=True,
    lr_scheduler_type="constant",
    dataloader_num_workers=1,  # Use multiple workers for data loading
    push_to_hub=False
)

trainer = SFTTrainer(
    model=base_model,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=training_args, max_seq_length=512  # Directly pass max_seq_length here if needed
)

#for param in base_model.model.base_model.parameters():
#    param.requires_grad = False
    
trainer.train()
trainer.model.save_pretrained(output_dir)