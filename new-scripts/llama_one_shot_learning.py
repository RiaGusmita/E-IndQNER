import json
import time
import logging
from transformers import logging as hf_logging
import torch
import time
from transformers import BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
import torch
import re
import ast
import pandas as pd
import ast
from accelerate import Accelerator
from collections import deque
from rdflib import Graph
import json
from tqdm import tqdm
import chardet
import glob

def generate_answer(pipe, example):
    
    prompt = pipe.tokenizer.apply_chat_template(example["messages"][:2],
                                                tokenize=False,
                                                add_generation_prompt=True)
    terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
    
    outputs = pipe(prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1
                )
    generated_text = outputs[0]['generated_text'][len(prompt):]
    return {"content": example["messages"][1]['content'], "generated_text": generated_text}

def create_input_prompt(system_message, user_prompt):
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": user_prompt},
        ]
    }
#base_model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
#base_model_name = "meta-llama/Llama-3.1-70B-Instruct"
base_model_name = "results/llama-3.1-70B/checkpoint-1247"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device_map = "auto"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(  ## If it fails at this line, restart the runtime and try again.
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True,
    low_cpu_mem_usage=True
)
model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
model.config.pretraining_tp = 1

# Specify the directory and file pattern
file_pattern = '../new-dataset-quran/*'  # Use '*' for all files, or specify a pattern like '*.txt'

# Retrieve all matching filenames
#chapter_list = glob.glob(file_pattern)


chapter_list = ["100. al-_Adiyat_normalized.txt"]

for chapter in chapter_list:
    chapter = f"../new-dataset-quran/{chapter}"
    print(chapter)
    with open(chapter, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    f.close()
    
    with open(chapter, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    f.close()
    
    for sentence in sentences:
        print(sentence.strip())

    # Open and read the JSON file
    with open("named_entity_class_dictionary.json", "r") as json_file:
        named_entity_classes_dict = json.load(json_file)
    
    # Print the loaded dictionary
    print(named_entity_classes_dict)
    named_entity_classes = [named_entity_class for named_entity_class in named_entity_classes_dict]
    print(named_entity_classes)
    
    chapter_dict = dict()
    num = 0
    chapter_id = chapter.split("/")[-1].split(".")[0]
    verses = dict()
    for sentence in sentences:
        num +=1
        system_message = f"""
         Given the following entity classes and sentences, label entity mentions with their respective classes in sentences according to the sentences' context. 
         In the output, only include entity mentions and their respective class in the given output format. No needed further explanation.
         CONTEXT: entity classes: {named_entity_classes}. 
         Example sentence: Jika kamu (tetap) dalam keraguan tentang apa (Al-Qur’an) yang Kami turunkan kepada hamba Kami (Nabi Muhammad), buatlah satu surah yang semisal dengannya dan ajaklah penolong-penolongmu selain Allah, jika kamu orang-orang yang benar.
         Example output: Jika/O kamu/O (/O tetap/O )/O dalam/O keraguan/O tentang/O apa/O (/O Al-Qur’an/HolyBook )/O yang/O Kami/O turunkan/O kepada/O hamba/O Kami/O (/O Nabi/O Muhammad/Messenger )/O ,/O buatlah/O satu/O surah/O yang/O semisal/O dengannya/O dan/O ajaklah/O penolong-penolongmu/O selain/O Allah/Allah ,/O jika/O kamu/O orang-orang/O yang/O benar/O ./O
        """
        user_prompt = f"""
        Test sentence: {sentence}
        Test output:
        """
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer) 
        # Generate answers for the dataset
        message = create_input_prompt(system_message, user_prompt)
        results = generate_answer(pipe, message)
        outputs = results["generated_text"]
        select_outputs = outputs
        
        print("##############")
        print(num)
        print(select_outputs)
        print("###############")
        verses[num] = {
            "chapterid": chapter_id,
            "verse_id": num,
            "verse": {"id": sentence.strip()},
            "labels": {"id": select_outputs}
        }
    chapter_dict[chapter_id] = verses
    
    # Save the dictionary to a JSON file
    with open(f"../al-quran-dataset-formatted/chapter_{chapter_id}.json", "w") as json_file:
        json.dump(chapter_dict, json_file, indent=4)  # 'indent' adds formatting for readability