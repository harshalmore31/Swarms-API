import json
import os
from time import sleep

import requests
from dotenv import load_dotenv

load_dotenv()

# Base URL for local testing
BASE_URL = "http://localhost:8080"

# Test API key - you'll need to replace this with a valid key
API_KEY = os.getenv("SWARMS_API_KEY")

# Headers used for all requests
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def check_server_running():
    """Check if the API server is running"""
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
        return True
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server!")
        print("\nPlease start the server first with:")
        print("\nuvicorn api.api:app --host 0.0.0.0 --port 8080 --reload")
        print("\nMake sure you're in the project root directory when running this command.")
        return False


def test_run_swarm():
    """Test running a swarm endpoint"""
    payload = {
        "name": "Test Swarm",
        "description": "A test swarm",
        "agents": [
            {
                "agent_name": "Research Agent",
                "description": "Conducts research",
                "system_prompt": "You are a research assistant.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
            },
            {
                "agent_name": "Writing Agent",
                "description": "Writes content",
                "system_prompt": "You are a content writer.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "Write a short blog post about AI agents.",
    }

    print("\n=== Run Swarm Test ===")
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )
    # print(f"Status Code: {response.status_code}")
    # print(f"Response: {response.json()}")

    output = response.json()

    return json.dumps(output, indent=4)


def test_batch_swarm():
    """Test the batch swarm completions endpoint"""
    payload = [
        {
            "name": "Batch Swarm 1",
            "description": "First swarm in the batch",
            "agents": [
                {
                    "agent_name": "Research Agent",
                    "description": "Conducts research",
                    "system_prompt": "You are a research assistant.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1,
                },
                {
                    "agent_name": "Analysis Agent",
                    "description": "Analyzes data",
                    "system_prompt": "You are a data analyst.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1,
                },
            ],
            "max_loops": 1,
            "swarm_type": "SequentialWorkflow",
            "task": "Research AI advancements.",
        },
        {
            "name": "Batch Swarm 2",
            "description": "Second swarm in the batch",
            "agents": [
                {
                    "agent_name": "Writing Agent",
                    "description": "Writes content",
                    "system_prompt": "You are a content writer.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1,
                },
                {
                    "agent_name": "Editing Agent",
                    "description": "Edits content",
                    "system_prompt": "You are an editor.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1,
                },
            ],
            "max_loops": 1,
            "swarm_type": "SequentialWorkflow",
            "task": "Write a summary of AI research.",
        },
    ]

    print("\n=== Batch Swarm Test ===")
    response = requests.post(
        f"{BASE_URL}/v1/swarm/batch/completions",
        headers=headers,
        json=payload,
    )
    print(f"Status Code: {response.status_code}")
    # print(f"Response: {response.json()}")

    return json.dumps(response.json(), indent=4)


def print_logs():
    """Print the logs for the swarm"""
    response = requests.get(f"{BASE_URL}/v1/swarm/logs", headers=headers)
    print(response.status_code)
    return json.dumps(response.json(), indent=4)


def run_all_tests():
    """Run all tests with some delay between them"""
    # Test basic health endpoint
    check_server_running()
    sleep(1)

    # # Test running a swarm
    # print(test_run_swarm())

    # print(test_batch_swarm())

    # Print the logs
    print(print_logs())


if __name__ == "__main__":
    print("Starting API tests...")
    run_all_tests()
    print("\nTests completed!")
