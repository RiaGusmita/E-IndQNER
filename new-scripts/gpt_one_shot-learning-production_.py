import pandas as pd
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm
import json
import ast
# Initialize OpenAI client
client = OpenAI(api_key="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")

def query_llm(prompt):
    """
    Send a prompt to the LLM and return the response.
    """
    print(prompt)
    response = client.chat.completions.create(
        messages=prompt,
        model="ft:gpt-4o-mini-2024-07-18:personal:e-indqner:AJ44A6B4" #change the model to model="gpt-3.5-turbo if you want to use gpt-3.5 
    )
    return {"response": response.choices[0].message.content}

def read_first_few_lines(file_path, num_lines=0):
    lines = []
    try:
        with open(file_path, 'r') as file:
            if num_lines==0:
                lines = file.readlines()
            else:
                for _ in range(num_lines):
                    lines.append(file.readline().strip())
    except Exception as e:
        return str(e)
    return lines


# Read and display the first few lines of the file
domain = "specific-domain"
file_path = f"../new-datasets/gpt-test.jsonl"
first_few_lines = read_first_few_lines(file_path)
chapter = 114
predict_dict = dict()
num = 1
for line in tqdm(first_few_lines):
    line = ast.literal_eval(line)
    results = query_llm(line["content"]["messages"])
    predict_dict[num]=results['response']
    num +=1

# Save the dictionary to a JSON file
with open("results-zeroshot-attempt-1-gpt.json", "w") as json_file:
    json.dump(predict_dict, json_file, indent=4)  # 'indent' adds formatting for readability

print("Results saved")