
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
# Modify BASE_URL as needed. For local testing, use localhost.
BASE_URL = "http://localhost:8080"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()

def run_single_swarm():
    # Read the document to be ingested for RAG.
    try:
        with open("data/markmanson.md", "r", encoding="utf-8") as file:
            content = file.read()
        print("Successfully read markmanson.md file.")
    except Exception as e:
        print(f"Error reading file: {e}")
        content = "Default sample text for RAG ingestion."

    payload = {
        "name": "Philosopher Swarm",
        "description": "A swarm of philosopher agents collaboratively exploring the meaning of life using Retrieval-Augmented Generation.",
        "agents": [
            {
                "agent_name": "Socratic Inquirer",
                "description": "Poses probing questions and challenges assumptions in true Socratic style.",
                "system_prompt": "You are a Socratic Inquirer who asks insightful questions and encourages deep philosophical exploration.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7,
                "auto_generate_prompt": False,
                "rag_collection": "philosophy_collection",
                "rag_documents": [content]
            },
            {
                "agent_name": "Aristotelian Analyst",
                "description": "Analyzes ideas with logical rigor and empirical reasoning reminiscent of Aristotle.",
                "system_prompt": "You are an Aristotelian analyst who uses logic and evidence to discuss philosophical concepts.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.6,
                "auto_generate_prompt": False,
                "rag_collection": "philosophy_collection1",
                "rag_documents": []  # No additional text; will use already indexed data.
            },
            {
                "agent_name": "Existential Critic",
                "description": "Questions the nature of existence and challenges traditional views about meaning.",
                "system_prompt": "You are an existential critic who delves into the uncertainties of life and existence.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.8,
                "auto_generate_prompt": False,
                "rag_collection": "philosophy_collection2",
                "rag_documents": []  # Uses the same collection for RAG search.
            }
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
        "task": "Discuss the meaning of life and analyze various philosophical viewpoints using the provided textual corpus.",
        "output_type": "dict"
    }

    print("Sending swarm request...")
    response = requests.post(f"{BASE_URL}/v1/swarm/completions", headers=headers, json=payload)
    print("Status Code:", response.status_code)
    try:
        result = response.json()
    except Exception as e:
        result = {"error": f"Failed to parse JSON response: {e}"}
    return result

def get_logs():
    response = requests.get(f"{BASE_URL}/v1/swarm/logs", headers=headers)
    try:
        return response.json()
    except Exception as e:
        return {"error": f"Failed to retrieve logs: {e}"}

if __name__ == "__main__":
    # Run health check
    health = run_health_check()
    print("Health Check:")
    print(json.dumps(health, indent=4))
    
    # Run a single swarm with RAG capabilities
    swarm_result = run_single_swarm()
    print("Swarm Result:")
    print(json.dumps(swarm_result, indent=4))
    
    # Retrieve and print logs
    logs = get_logs()
    print("Logs:")
    print(json.dumps(logs, indent=4))

