import os
import pandas as pd
os.environ["OPENAI_API_KEY"]=""

from biomni.agent import A1
import biomni
print(biomni)

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='gpt-4o', use_tool_retriever=True)

import json
results = []
with open("../../rarebenchdata/mygene2_test_gpt-5_top5.jsonl", 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
cl = 'mygene2'
disease_num = "one"

df_disease = pd.read_csv("../../rarebenchdata/mygene2_data_diseasenrihced.csv")

prompt_example = '''Consider you are a genetic counselor. {PhenotypeList}. The diagnosis result is {DiseaseInfo}. Can you suggest a list of {TopK} possible causal genes? Please return gene symbols as a comma separated list. Example: 'ABC1, BRAC2, BRAC1' or “not applicable” if you cannot provide the result.'''
output_data = []
from tqdm import tqdm
for index in tqdm(range(len(results))):
    try:
        prompt = results[index]['body']['messages'][1]['content'].split('. ')[1]
        prompt = prompt_example.format(PhenotypeList = prompt, DiseaseInfo=df_disease.loc[index]["disease_name"], TopK='top 1')
#         print(prompt)
#         break
#         print(prompt)
        log = agent.go(prompt)
        output_data.append(log[0][-1])
    except:
        output_data.append("nan")
#     break

import pickle
with open(f"/home/tl688/project/{cl}_biomni_withdisease_withsearch_geneprioritize_top1.pkl", "wb") as file:
    pickle.dump(output_data, file)
        
df_save = pd.DataFrame()
df_save['output'] = output_data
df_save.to_csv(f"/home/tl688/project/{cl}_biomni_withdisease_withsearch_geneprioritize_top1.csv")