import json

# Load the JSON data
with open('TextToSQL_AdvNLP-ash/data/resdsql_pre/preprocessed_dataset_test.json') as json_file:
    data = json.load(json_file)

# Extract the first 100 questions
questions = [entry['sql'] for entry in data[:100]]

# Write the questions to a text file
with open('sql.txt', 'w') as file:
    for question in questions:
        file.write(f'{question}\n')
