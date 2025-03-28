import json
import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

# Standard headers for all requests
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def run_swarm(swarm_config):
    """Execute a swarm with the provided configuration."""
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions", headers=headers, json=swarm_config
    )
    return response.json()


def create_icd10_coding_swarm(clinical_documentation):
    """
    Create a swarm for ICD-10 coding assistance.

    Args:
        clinical_documentation (str): Clinical notes, discharge summary, or other medical documentation

    Returns:
        dict: The swarm execution result containing ICD-10 codes with justifications.
    """
    # Define specialized prompts for medical coding
    DOCUMENTATION_ANALYZER_PROMPT = """
    You are a clinical documentation specialist. Your role is to carefully analyze medical documentation to identify all documented medical conditions, procedures, and relevant clinical factors.

    Your tasks include:
    1. Identifying all medical conditions explicitly stated in the documentation
    2. Noting the specificity of each documented condition (e.g., type, acuity, severity, etiology)
    3. Identifying any documented causal relationships between conditions
    4. Recognizing documented complications and manifestations
    5. Distinguishing between confirmed diagnoses and suspected/ruled-out conditions
    6. Identifying any documented procedures or interventions

    When analyzing documentation:
    - Focus only on explicitly documented conditions (do not infer diagnoses)
    - Note when documentation lacks necessary specificity for precise coding
    - Identify clinical indicators and findings that support documented diagnoses
    - Organize conditions by system or by primary/secondary status
    - Highlight any areas where provider clarification might be needed
    """

    ICD10_CODER_PROMPT = """
    You are an expert medical coder with extensive experience in ICD-10-CM and ICD-10-PCS coding.
    Your responsibility is to translate clinical documentation into accurate diagnostic and procedural codes.

    Your tasks include:
    1. Analyzing clinical documentation for relevant diagnoses and procedures
    2. Assigning appropriate ICD-10-CM codes for all documented conditions
    3. Identifying the principal diagnosis and secondary diagnoses
    4. Ensuring documentation supports the specificity required for accurate coding
    5. Following correct coding guidelines, including combination codes, excludes notes, and sequencing rules

    When coding a case:
    - Identify all documented conditions requiring codes
    - Use the highest level of specificity available in the documentation
    - Follow ICD-10-CM Official Guidelines for Coding and Reporting
    - Identify potentially missing documentation needed for complete coding
    - Distinguish between symptoms and confirmed diagnoses
    - Recognize when "unspecified" codes are appropriate vs. when more specific documentation is needed

    Follow these principles:
    - Code only what is explicitly documented by the provider
    - Do not infer diagnoses not stated in the documentation
    - Use causal relationships (due to, secondary to) to guide code selection
    - Apply conventions for combination codes, manifestation codes, and sequencing
    - Identify conditions that affect MS-DRG or risk adjustment
    """

    CODING_VALIDATOR_PROMPT = """
    You are a senior coding auditor with extensive experience in ICD-10-CM coding validation. Your role is to review proposed code assignments for accuracy, specificity, and compliance with coding guidelines.

    Your responsibilities include:
    1. Validating that assigned codes accurately represent documented conditions
    2. Ensuring codes have the highest level of specificity supported by documentation
    3. Verifying correct application of coding guidelines and conventions
    4. Checking proper sequencing of codes (principal diagnosis, CC/MCC, etc.)
    5. Identifying any coding errors or documentation gaps

    When reviewing code assignments:
    - Verify that each code matches the documented condition
    - Check for missing required additional codes (manifestation codes, combination requirements)
    - Verify exclusion terms are properly addressed
    - Validate that sequencing follows official guidelines
    - Confirm that codes are supported by explicit documentation
    - Identify any opportunities for query or documentation improvement

    Follow these principles:
    - Reference specific ICD-10-CM guidelines when identifying issues
    - Note when a more specific code could be used with additional documentation
    - Distinguish between actual coding errors and documentation opportunities
    - Consider compliance risk areas in your review
    - Provide educational feedback on any identified issues
    """

    # Configure the ICD-10 coding swarm
    swarm_config = {
        "name": "ICD-10 Coding Assistant",
        "description": "A specialized swarm for accurate ICD-10 coding",
        "agents": [
            {
                "agent_name": "Documentation Analyzer",
                "description": "Reviews clinical documentation and identifies relevant conditions",
                "system_prompt": DOCUMENTATION_ANALYZER_PROMPT,
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
            },
            {
                "agent_name": "ICD-10-CM Coder",
                "description": "Assigns diagnosis codes based on documentation analysis",
                "system_prompt": ICD10_CODER_PROMPT,
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
            },
            {
                "agent_name": "Coding Validator",
                "description": "Reviews code assignments for accuracy, specificity, and guideline compliance",
                "system_prompt": CODING_VALIDATOR_PROMPT,
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
            },
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
        "task": f"""
        Analyze the following clinical documentation and provide accurate ICD-10-CM codes:

        {clinical_documentation}

        For each assigned code, provide:
        1. The ICD-10-CM code
        2. The corresponding condition/diagnosis
        3. Justification from the documentation
        4. Code sequencing rationale

        Also identify any areas where documentation could be improved for more accurate coding.
        """,
    }

    # Execute the swarm
    result = run_swarm(swarm_config)
    return result


def run_example_icd10_coding():
    """Run an example ICD-10 coding case."""

    # Example clinical documentation
    clinical_documentation = """
    DISCHARGE SUMMARY

    Patient: John Smith
    MRN: 12345678
    DOB: 01/15/1962
    Admission Date: 06/10/2023
    Discharge Date: 06/15/2023

    FINAL DIAGNOSES:
    1. Acute exacerbation of chronic systolic congestive heart failure
    2. Non-ST elevation myocardial infarction
    3. Type 2 diabetes mellitus with diabetic nephropathy, estimated GFR 42
    4. Hypertension
    5. Hyperlipidemia

    HISTORY OF PRESENT ILLNESS:
    The patient is a 61-year-old male with a history of congestive heart failure (EF 35% on prior echo 3 months ago) who presented to the emergency department with a 3-day history of worsening shortness of breath, orthopnea, and lower extremity edema. The patient reported chest pressure while walking to the bathroom. Initial troponin was elevated at 0.15 ng/mL, with subsequent rise to 0.25 ng/mL, consistent with NSTEMI. BNP was significantly elevated at 1,250 pg/mL.

    HOSPITAL COURSE:
    The patient was admitted to the cardiac care unit. He was treated with IV furosemide with good diuretic response, losing 3.5 liters of fluid during hospitalization. Cardiology was consulted and performed a left heart catheterization, which revealed 70% stenosis in the mid-LAD with PCI and drug-eluting stent placement. The patient's symptoms improved significantly with diuresis and coronary intervention.

    His diabetes was managed with insulin during the hospitalization, with blood glucose ranging from 130-210 mg/dL. Nephrology was consulted regarding his diabetic nephropathy and recommended adjustment of medications appropriate for his renal function.

    PROCEDURES:
    1. Left heart catheterization with percutaneous coronary intervention and drug-eluting stent placement to mid-LAD (06/12/2023)

    DISCHARGE MEDICATIONS:
    1. Lisinopril 10 mg daily
    2. Carvedilol 12.5 mg twice daily
    3. Spironolactone 25 mg daily
    4. Furosemide 40 mg twice daily
    5. Atorvastatin 80 mg daily
    6. Aspirin 81 mg daily
    7. Clopidogrel 75 mg daily
    8. Insulin glargine 20 units nightly
    9. Insulin lispro sliding scale before meals

    DISCHARGE PLAN:
    1. Follow-up with cardiology in 2 weeks
    2. Follow-up with primary care physician in 1 week
    3. Follow-up with nephrology in 3 weeks
    4. Daily weights, 2 gram sodium diet, fluid restriction to 1.5 liters daily
    """

    # Run the ICD-10 coding swarm
    result = create_icd10_coding_swarm(clinical_documentation)

    # Print and save the result
    print(json.dumps(result, indent=4))
    return result


if __name__ == "__main__":
    run_example_icd10_coding()
