from dotenv import load_dotenv
import os
import json
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import subprocess
import re
import sys
load_dotenv()


# load the json object and create a dictionary for questions and their corresponding dataset
json_file_path = 'TextToSQL_AdvNLP-ash/data/resdsql_pre/preprocessed_dataset_test.json'
with open(json_file_path, 'r') as f:
    data = json.load(f)
db_questions = {item['norm_question']: item['db_id'] for item in data}


# change .sqlite/ or any file extension to .db (because langchain is weird and wont take any other extension other than .db)
def change_extension(folder_path, old_ext, new_ext):
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            # Check the file extension and if it is .sqlite then rename it
            if filename.endswith(old_ext):
                old_file = os.path.join(foldername, filename)
                new_file = os.path.join(foldername, os.path.splitext(filename)[0] + new_ext)
                os.rename(old_file, new_file)
    print("File extensions changed successfully.")
change_extension('/spider/database', '.sqlite', '.db')


'''THE CODE BELOW USES open source models(gpt4all-l13b-snoozy and vicuna-13b-1.1-q4.2) that are downloaded locally'''
local_path = '/Users/dishankj/Library/Application Support/nomic.ai/GPT4All/ggml-gpt4all-l13b-snoozy.bin'
# local_path = 'ggml-vicuna-13b-1.1-q4_2.bin'
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, n_ctx=512, n_threads=8)

# To use OpenAI models use the code below
# llm=OpenAI(temperature=0)


counter = 0
for question, db_id in db_questions.items():
    if counter >= 100:
        break

    dburl = f"sqlite:///spider/database/{db_id}/{db_id}.db"
    db = SQLDatabase.from_uri(dburl)
    toolkit = SQLDatabaseToolkit(db=db,llm=llm)
    # print(f"dburl: {dburl}, question: {question}")
    try:
        agent_executor = create_sql_agent(llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, n_ctx=1024, n_threads=8),toolkit=toolkit,verbose=True)
        agent_executor.run(question)
        # langchain_output.append(agent_executor.langchain_output)
    except Exception as e:
        print(f"An error occurred: {e}")
        continue
    
    counter += 1







