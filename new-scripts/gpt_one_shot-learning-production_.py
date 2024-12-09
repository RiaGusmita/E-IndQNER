import pandas as pd
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm
import json
import ast
import glob
# Initialize OpenAI client
client = OpenAI(api_key="<open ai key>")

def query_llm(prompt):
    """
    Send a prompt to the LLM and return the response.
    """
    print(prompt)
    response = client.chat.completions.create(
        messages=prompt,
        model="ft:gpt-4o-2024-08-06:personal:e-indqner:Aa8Nj3eM" #change the model to model="gpt-3.5-turbo if you want to use gpt-3.5 
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
#domain = "specific-domain"
#file_path = f"../new-datasets/new-gpt_test.jsonl"
#first_few_lines = read_first_few_lines(file_path)
#print(first_few_lines)

# Open and read the JSON file
with open("named_entity_class_dictionary.json", "r") as json_file:
    named_entity_classes_dict = json.load(json_file)
    
# Print the loaded dictionary
named_entity_classes = [named_entity_class for named_entity_class in named_entity_classes_dict]
#chapter_list = ["5. al-Ma_idah_normalized.txt", "6. al-An_am_normalized.txt", "4. an-Nisa__normalized.txt"]
# Specify the directory and file pattern
file_pattern = '../new-dataset-quran/*'  # Use '*' for all files, or specify a pattern like '*.txt'

# Retrieve all matching filenames
chapter_list = glob.glob(file_pattern)

for chapter in chapter_list:
    chapter_name = chapter.split("/")[-1]
    chapter_id = chapter_name.split(".")[0]
    if len(chapter_id)==1:
        chapter_id = f"00{chapter_id}"
    elif len(chapter_id)==2:
        chapter_id = f"0{chapter_id}"
    if chapter_id == "001" or chapter_id == "002" or chapter_id=="003" or chapter=="004":
        continue
    print("--------------------")
    print(chapter_name)
    print("--------------------")
    #chapter = f"../new-dataset-quran/{chapter}"
    with open(chapter, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    f.close()
    num = 1
    chapter_dict = dict()
    verses = dict()
    for sentence in sentences:
        sentence = sentence.strip()
        messages = [
            {"content": "\n    Given the following entity classes, label entity mentions with their respective classes in sentences according to the sentences' context. \n    In the output, only include entity mentions and their respective class in the given output format. No needed further explanation.\n    CONTEXT: entity classes: ['O', 'HolyBook', 'Allah', 'Throne', 'AfterLifeEvent', 'CalendarEvent', 'HistoricEvent', 'PhysicalEvent', 'AstronomicalBody', 'Messenger', 'Prophet', 'Angel', 'ChildrenOfAdam', 'HistoricPeople', 'HistoricPerson', 'Queen', 'King', 'Food', 'Fruit', 'Color', 'Bird', 'Plant', 'Disease', 'Mosque', 'Weaponary', 'Religion', 'GeographicalLocation', 'AfterlifeLocation', 'Language', 'FalseDeity', 'Idol'].\n    ", "role": "system"},
            {"content": f"\n    {sentence}\n    Test output:\n    ", "role": "user"}
        ]
        results = query_llm(messages)
        verses[num] = {
                "chapterid": chapter_id,
                "verse_id": num,
                "verse": {"id": sentence.strip()},
                "labels": {"id": results['response']}
        }
        num +=1
    chapter_dict[chapter_id] = verses
    
    # Save the dictionary to a JSON file
    with open(f"outputs/chapter_{chapter_id}.json", "w") as json_file:
        json.dump(chapter_dict, json_file, indent=4)  # 'indent' adds formatting for readability
    