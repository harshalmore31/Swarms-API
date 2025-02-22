import os
import requests
from time import sleep
import json

# Base URL for local testing
BASE_URL = "http://localhost:8080"

# Test API key - you'll need to replace this with a valid key
API_KEY = os.getenv("SWARMS_API_KEY")

# Headers used for all requests
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def test_health():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("\n=== Health Check Test ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


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
            },
            {
                "agent_name": "Writing Agent",
                "description": "Writes content",
                "system_prompt": "You are a content writer.",
                "model_name": "gpt-4o",
                "role": "worker",
            },
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
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


def run_all_tests():
    """Run all tests with some delay between them"""
    try:
        # Test basic health endpoint
        test_health()
        sleep(1)

        # Test running a swarm
        print(test_run_swarm())

    except requests.exceptions.ConnectionError:
        print(
            "\nERROR: Could not connect to the API. Make sure the API server is running on localhost:8080"
        )
    except Exception as e:
        print(f"\nERROR: An error occurred during testing: {str(e)}")


if __name__ == "__main__":
    print("Starting API tests...")
    run_all_tests()
    print("\nTests completed!")
