


# A Versatile Multi-Modal Agent for Disease Diagnosis and Risk Gene Prioritization

## Overview


Hygieia is designed to make diagnosis and predict risk gene based on AI agent. Our framework mainly relies on Biomni. Therefore, we refer their tutorial to get start.


## Quick Start

### Installation

Our software environment is massive and we provide a single setup.sh script to setup.
Follow this [file](biomni_env/README.md) to setup the env first.

Then activate the environment E1:

```bash
conda activate biomni_e1
```

then install the biomni official pip package:

```bash
pip install biomni --upgrade
```

For the latest update, install from the github source version, or do:

```bash
pip install git+https://github.com/snap-stanford/Biomni.git@main
```

Lastly, configure your API keys using one of the following methods:


### API setup

The users need API access for different models. We support all open/closed source LLMs as backbone. As an example, we provide an Azure implementation of general API, recorded in **biomni/tool/literature.py**. Please also define the variables based on the following API list:

```env


# Required: Anthropic API Key for Claude models, as well as Azure API keys.
ANTHROPIC_API_KEY=your_anthropic_api_key_here
subscription_key= your subscription
endpoint=your endpoint
base_url=your base url

# Optional: OpenAI API Key (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Azure OpenAI API Key (if using Azure OpenAI models)
OPENAI_API_KEY=your_azure_openai_api_key
OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

# Optional: AI Studio Gemini API Key (if using Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: groq API Key (if using groq as model provider)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Set the source of your LLM for example:
#"OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"
LLM_SOURCE=your_LLM_source_here

# Optional: AWS Bedrock Configuration (if using AWS Bedrock models)
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_REGION=us-east-1

# Optional: Custom model serving configuration
# CUSTOM_MODEL_BASE_URL=http://localhost:8000/v1
# CUSTOM_MODEL_API_KEY=your_custom_api_key_here

# Optional: Biomni data path (defaults to ./data)
# BIOMNI_DATA_PATH=/path/to/your/data

# Optional: Timeout settings (defaults to 600 seconds)
# BIOMNI_TIMEOUT_SECONDS=600
```

## Tutorial
Please refer the folder **tutorial** for details, and you need to setup OpenAI API for accessing, and downloading the data from [rarebench](https://huggingface.co/datasets/chenxz/RareBench) for testing.


## Cite Us

```
tbd
```
