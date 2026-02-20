import os
from biomni.tool.literature import run_diagnosis
from biomni.tool.literature import *

from openai import AzureOpenAI
from openai import OpenAI
def pattern_match(text,p1="<answer>",p2="</answer>"):
    start_marker = p1
    end_marker = p2

    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    try:
        if start_index != -1 and end_index != -1:
            # Adjust start_index to begin *after* the start_marker
            extracted_text = text[start_index + len(start_marker) : end_index]
    except:
        extracted_text = 'nan'
    return extracted_text


def run_diagnosis_verifier(input_patient_information, knowinte=True, knowgene=None, context_info=None, geneinfo = None):
    """
    Run diagnosis based on the multiple agent system.

    Parameters:
    -----------
    input_patient_information : str
        The information of this patient.

    Returns:
    --------
    str
        Diagnosis result
    """
    epoch = 5
    endpoint = "https://75244-mfztkr7x-eastus2.cognitiveservices.azure.com/"
    model_name = "gpt-5-chat-3"
    model = "gpt-5-chat-3"

    subscription_key = "8PNMdsUYGdMPsCfl0baO0hjtnGE2m40zJTrUGC3vKnHdpjnkOgeQJQQJ99BIACHYHv6XJ3w3AAAAACOG7VZI"
    api_version = "2025-03-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
#     client_verifier = OpenAI(
#     base_url="https://75244-mfztkr7x-eastus2.services.ai.azure.com/openai/v1/",
#     api_key=subscription_key)
    prior_knowledge = ""
    gene_knowledge = ""
    if knowinte:
#         if context_info != "Research Information: No results found on Google Scholar.Google Information: PubMed Information: No papers found on PubMed after multiple query attempts.":
#             prior_knowledge += context_info
        pheno_list = input_patient_information.split(", ")
        for key in pheno_list:
            key = key.replace(".","")
            response = client.responses.create(
                model=model,
                input=f"Please summarize the content in one paragraph: {matched_pheno_dict[key]}",
                temperature=0.1
            )
            modelout_response_summary = response.output_text
            prior_knowledge += f"{key}:{modelout_response_summary}"
        
    
    if knowgene != None:
        prior_knowledge += "Risk Gene Information: The detected risk gene is: " + knowgene +"."
    if geneinfo!=None and geneinfo!='nan':
        prior_knowledge += "The function of this gene is: "+geneinfo
        
    disease_num = "one"
    meta_prompt = f"You are a knowledgeable biologist and physician who has been provided with the following context information: {prior_knowledge}. The input phenotypes are: {input_patient_information}. Based on this input, your task is to carefully analyze the details and produce a diagnosis by identifying the top {disease_num} most relevant disease names. These disease names should be standardized according to the conventions used in OMIM, Orphanet, and CCRD. In completing this task, you should integrate the given context thoughtfully, document your reasoning process within <thinking></thinking>, and present the final list of the top {disease_num} disease names within <solution></solution>."
    response = client.responses.create(
        model=model,
        input=meta_prompt,
        temperature=0.1
    )
    modelout = response.output_text
#     modelout = 'nan'
    
    verifier_prompt = '''You are a helpful biologist and physician tasked with extracting and parsing the task output from an agent’s message history. Review the entire message history and verify the diagnostic process and final result against the provided context. The context information is {CONTEXT}, and the diagnosis process and conclusion are {OUTPUT}. Carefully integrate the context, think through the evidence, and produce a binary judgment: output YES if the diagnosis is correct, or NO if it is incorrect. Document your reasoning within <think></think>, and present your final judgment within <answer></answer>.'''
    for _ in range(epoch):
        print(f"starting self-verification: epoch {_}")
        try:
            response = client.responses.create(
                model=model,
                input=verifier_prompt.format(CONTEXT=prior_knowledge, OUTPUT=pattern_match(modelout, "<solution>", "</solution>")),
                temperature=0.1
            )
            modelout_verifier = response.output_text
    #         print(modelout_verifier)
            modelout_verifier = pattern_match(modelout_verifier)
        except:
            modelout_verifier = "NO"
        if "YES" in modelout_verifier:
            break
        else:
            response = client.responses.create(
                model=model,
                input=f"The previous diagnosis process: {modelout}, is wrong. Make new diagnosis again with thinking and searching."+meta_prompt,
                temperature=0.1
            )
            modelout = response.output_text
    time.sleep(5)
    return modelout
    
    

def run_generank(input_patient_information):
    pass