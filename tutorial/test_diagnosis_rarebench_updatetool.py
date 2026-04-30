import os
os.environ["OPENAI_API_KEY"]=""

from biomni.agent import A1
from biomni.tool.diagnosis import run_diagnosis
import biomni
print(biomni)


#run benchmark with the given dataset.
# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='gpt-4o', use_tool_retriever=True)
import json
results = []
with open("../../rarebenchdata/fourdata_test_o4-mini_top1.jsonl", 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
cl = 'fourdata'
disease_num = "one"

output_data = []
from tqdm import tqdm
for index in tqdm(range(len(results))):
    try:
        prompt = results[index]['body']['messages'][1]['content'].split('. ')[1] + '. ' + f'You should return the analysis process with format <thinking></thinking> and return the top {disease_num} disease cleaned name with format <solution><solution>.'
        log = agent.go(prompt)
        output_data.append(log[0][-1])
    except:
        output_data.append("nan")
#     break
import pandas as pd
df_save = pd.DataFrame()

df_save['output'] = output_data

df_save.to_csv(f"/home/tl688/project/{cl}_biomni_functioncall_hyedia_contextinfo_top{disease_num}.csv")