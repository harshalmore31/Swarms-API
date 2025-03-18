# tools - search, code executor, create api

import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


def run_single_swarm():
    payload = {
        "name": "Gold ETF Analysis Swarm",
        "description": "Swarm for analyzing gold ETFs",
        "agents": [
            {
                "agent_name": "Gold Market Analyst",
                "description": "Analyzes trends in the gold market and ETFs related to gold.",
                "system_prompt": "You are a financial analyst specializing in gold and gold ETFs.",
                "model_name": "groq/deepseek-r1-distill-llama-70b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
                "tools_dictionary": [],
            },
            {
                "agent_name": "Gold Economic Forecaster",
                "description": "Predicts economic trends affecting gold prices and ETFs.",
                "system_prompt": "You are an expert in economic forecasting with a focus on gold.",
                "model_name": "groq/deepseek-r1-distill-llama-70b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
                "tools_dictionary": [],
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best ETFs for investing in gold?",
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


def get_swarm_types():
    response = requests.get(f"{BASE_URL}/v1/swarms/available")
    print(response)
    return response.json()


if __name__ == "__main__":
    result = run_single_swarm()
    print("Swarm Result:")
    print(result)

    # swarm_types = get_swarm_types()
    # print("Swarm Types:")
    # print(swarm_types)

    # logs = get_logs()
    # logs = json.dumps(logs, indent=4)
    # print("Logs:")
