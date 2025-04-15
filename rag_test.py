import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
# Change BASE_URL if needed. For local testing, "http://localhost:8080" is used.
BASE_URL = "http://localhost:8080"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


def run_single_swarm():
    # Contextualized content specifically about Project Zenith.
    content1 = (
        "In 1987, a secret deep-space probe called Project Zenith was launched by an "
        "unknown coalition of scientists. The probe reportedly sent back signals from "
        "outside our solar system, detecting non-random patterns near Proxima Centauri. "
        "This groundbreaking project aimed to explore the potential for extraterrestrial "
        "life and advanced technologies. However, after sending transmissions until 1994, "
        "all official records of the project mysteriously disappeared, leading to numerous "
        "theories about its findings and the fate of the probe."
    )
    content2 = (
        "Project Zenith was not just a scientific endeavor; it represented humanity's "
        "first serious attempt to communicate with potential extraterrestrial civilizations. "
        "The probe's advanced technology allowed it to analyze cosmic signals, and its "
        "findings hinted at the existence of complex life forms beyond Earth. Despite its "
        "disappearance, the legacy of Project Zenith continues to inspire scientists and "
        "enthusiasts alike."
    )
    content3 = (
        "In the years following the launch of Project Zenith, various conspiracy theories "
        "emerged, suggesting that the probe had made contact with alien life. Some claimed "
        "that the project was shut down due to government pressure, while others believed "
        "it was a cover-up of groundbreaking discoveries. The mystery surrounding Project "
        "Zenith fuels ongoing debates in both scientific and popular culture."
    )

    # Swarm payload with agents configured for RAG functionality, focused on Project Zenith.
    payload = {
        "name": "Project Zenith Exploration Swarm",
        "description": "A swarm of agents designed to explore and analyze data from Project Zenith.",
        "agents": [
            {
                "agent_name": "Space Exploration Specialist",
                "description": "You are an expert in space exploration and extraterrestrial research.",
                "system_prompt": "You are a knowledgeable AI focused on space exploration.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7,
                "auto_generate_prompt": False,
                "rag_collection": "project_zenith_research",
                "rag_documents": [content1, content2, content3],
            },
            {
                "agent_name": "Extraterrestrial Communication Analyst",
                "description": "You specialize in analyzing signals and communications from potential extraterrestrial sources.",
                "system_prompt": "You are an AI focused on interpreting cosmic signals and their implications.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7,
                "auto_generate_prompt": False,
                "rag_collection": "project_zenith_research",
                "rag_documents": [content1, content2, content3],
            },
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
        "task": "Identify the key findings and implications of Project Zenith.",
        "output_type": "dict",
    }

    print("Sending swarm request...")
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions", headers=headers, json=payload
    )
    print("Status Code:", response.status_code)
    try:
        result = response.json()
    except Exception as ex:
        print("Error parsing JSON response:", ex)
        print("Response Text:", response.text)
        result = None
    return result


if __name__ == "__main__":
    # Run health check
    health = run_health_check()
    print("Health Check Response:")
    print(json.dumps(health, indent=4))

    # Run a single swarm with RAG-enabled agents
    swarm_result = run_single_swarm()
    print("Swarm Result:")
    print(json.dumps(swarm_result, indent=4))
