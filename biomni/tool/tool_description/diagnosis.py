description = [
    {
        "description": "This is a function for disease diagnosis and all quires related to disease diagnosis should follow this function. Making diagnosis for disease and genetic disorder based on the input patient information, including phenotypes, gene (optional), and other modalities for common and rare disease diagnosis.",
        "name": "run_diagnosis",
        "optional_parameters": [],
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
        "description": "Providing a list of possible causal genes for the given phenotypes and disease diagnosis based on the patient information.",
        "name": "run_generank",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Patient information which can be used for gene ranking",
                "name": "input_patient_information",
                "type": "str",
            }
        ],
    },
]
