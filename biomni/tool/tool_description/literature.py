description = [
    {
        "description": "Fetches supplementary information for a paper given its DOI "
        "and saves it to a specified directory.",
        "name": "fetch_supplementary_info_from_doi",
        "optional_parameters": [
            {
                "default": "supplementary_info",
                "description": "Directory to save supplementary files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The paper DOI",
                "name": "doi",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query arXiv for papers based on the provided search query.",
        "name": "query_arxiv",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query Google Scholar for papers based on the provided search "
        "query and return the first search result.",
        "name": "query_scholar",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query PubMed for papers based on the provided search query.",
        "name": "query_pubmed",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Maximum number of retry attempts with modified queries.",
                "name": "max_retries",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Search using Google search and return formatted results.",
        "name": "search_google",
        "optional_parameters": [
            {
                "default": 3,
                "description": "Number of results to return",
                "name": "num_results",
                "type": "int",
            },
            {
                "default": "en",
                "description": "Language code for search results",
                "name": "language",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'The search query (e.g., "protocol text or search question")',
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Extract the text content of a webpage using requests and BeautifulSoup.",
        "name": "extract_url_content",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Webpage URL to extract content from",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Extract text content from a PDF file.",
        "name": "extract_pdf_content",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "URL of the PDF file",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.",
        "name": "advanced_web_search_claude",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Maximum number of searches",
                "name": "max_searches",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Maximum number of retry attempts with modified queries.",
                "name": "max_retries",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "This is a function for disease diagnosis and all quires related to disease diagnosis should follow this function. Making diagnosis for disease and genetic disorder based on the input patient information, including phenotypes, gene (optional), and other modalities for common and rare disease diagnosis.",
        "name": "run_diagnosis",
        "optional_parameters": [            {
                "default": True,
                "description": "Whether we want to integrate external knowledge in the decision process.",
                "name": "knowinte",
                "type": "str",
            },
                    {
                "default": None,
                "description": "Whether we know the genes recommended for testing or not.",
                "name": "knowgene",
                "type": "str",
            },
                    {
                "default": None,
                "description": "Whether we know the functional information of this gene or not.",
                "name": "geneinfo",
                "type": "str",
            },
                    {
                "default": "one",
                "description": "Number of returned disease.",
                "name": "disease_num",
                "type": "str",
            },
                            {
                "default": "gpt-5.4-mini",
                "description": "Large Language Model used for generating disease diagnosis result.",
                "name": "model",
                "type": "str",
            },
                            {
                "default": "gpt-5.4-mini",
                "description": "Large Language Model used for generating verification result.",
                "name": "verifier_model",
                "type": "str",
            },
                            {
                "default": 3,
                "description": "Maximal iteration time for verification loop",
                "name": "max_verify_rounds",
                "type": "int",
            }],
        "required_parameters": [
            {
                "default": None,
                "description": "Patient information which can be used for diagnosis",
                "name": "input_patient_information",
                "type": "str",
            }
        ],
    },
    {
        "description": "Providing a list of possible genes for testing based on the given phenotypes as the patient information.",
        "name": "run_generank",
        "optional_parameters": [            {
                "default": True,
                "description": "Whether we want to integrate external knowledge in the decision process.",
                "name": "knowinte",
                "type": "str",
            },
                    {
                "default": None,
                "description": "Whether we know the disease or risk information of this patient.",
                "name": "knowdisease",
                "type": "str",
            },
                    {
                "default": "one",
                "description": "Number of returned gebes.",
                "name": "gene_num",
                "type": "str",
            },
                            {
                "default": "gpt-5.4-mini",
                "description": "Large Language Model used for generating disease diagnosis result.",
                "name": "model",
                "type": "str",
            },
                            {
                "default": "gpt-5.4-mini",
                "description": "Large Language Model used for generating verification result.",
                "name": "verifier_model",
                "type": "str",
            },
                            {
                "default": 0.0,
                "description": "Pause time for verification loop",
                "name": "sleep_sec",
                "type": "float",
            },                            
            {
                "default": 3,
                "description": "Maximal iteration time for verification loop",
                "name": "max_verify_rounds",
                "type": "int",
            }],
        "required_parameters": [
            {
                "default": None,
                "description": "Patient information which can be used for generating gene ranking",
                "name": "input_patient_information",
                "type": "str",
            }
        ],
    },
]