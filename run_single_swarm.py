# tools - search, code executor, create api

import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "http://localhost:8080"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


def run_single_swarm():
    payload = {
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market trends",
                "system_prompt": "You are a financial analyst expert.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
            },
            {
                "agent_name": "Economic Forecaster",
                "description": "Predicts economic trends",
                "system_prompt": "You are an expert in economic forecasting.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best etfs and index funds for ai and tech?",
        "output_type": "dict",
    }

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    print(response)
    print(response.status_code)
    # return response.json()
    output = response.json()

    return json.dumps(output, indent=4)


def get_logs():
    response = requests.get(f"{BASE_URL}/v1/swarm/logs", headers=headers)
    output = response.json()
    # return json.dumps(output, indent=4)
    return output


if __name__ == "__main__":
    result = run_single_swarm()
    print("Swarm Result:")
    print(result)

    # logs = get_logs()
    # logs = json.dumps(logs, indent=4)
    # print("Logs:")
    # print(logs)
