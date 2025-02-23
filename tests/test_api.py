import requests
import json
from time import sleep
from typing import Dict, Any, Optional

# Base URL for local testing
BASE_URL = "http://localhost:8080"

# Test API key - replace with your valid key from .env
API_KEY = "sk-855b33490139c9cb0945692af14ef1cfae701a8b82ed1035f0302f07298cb4f9"

# Headers used for all requests
headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def make_request(method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Make an HTTP request and handle common errors
    """
    url = f"{BASE_URL}{endpoint}"
    try:
        if method.lower() == "get":
            response = requests.get(url, headers=headers)
        else:
            response = requests.post(url, headers=headers, json=json_data)
        
        # Try to get JSON response
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError:
            result = {"error": f"Invalid JSON response: {response.text}"}
            
        return {
            "status_code": response.status_code,
            "success": 200 <= response.status_code < 300,
            "data": result
        }
        
    except requests.exceptions.ConnectionError:
        return {
            "status_code": 503,
            "success": False,
            "data": {"error": f"Could not connect to {url}. Is the server running?"}
        }
    except Exception as e:
        return {
            "status_code": 500,
            "success": False,
            "data": {"error": str(e)}
        }

def test_health():
    """Test the health check endpoint"""
    print("\n=== Health Check Test ===")
    result = make_request("GET", "/health")
    print(f"Status Code: {result['status_code']}")
    print(f"Response: {result['data']}")
    return result['success']

def test_run_swarm():
    """Test running a swarm"""
    print("\n=== Run Swarm Test ===")
    
    # Test payload
    payload = {
        "name": "Test Swarm",
        "description": "A test swarm",
        "agents": [
            {
                "agent_name": "Research Agent",
                "description": "Conducts research",
                "system_prompt": "You are a research assistant.",
                "model_name": "gpt-4",
                "role": "researcher",
                "max_loops": 1,
                "temperature": 0.7
            },
            {
                "agent_name": "Writing Agent",
                "description": "Writes content",
                "system_prompt": "You are a content writer.",
                "model_name": "gpt-4",
                "role": "writer",
                "max_loops": 1,
                "temperature": 0.7
            }
        ],
        "max_loops": 1,
        "swarm_type": "sequential",
        "flow": "sequential",
        "task": "Write a short blog post about AI agents."
    }

    result = make_request("POST", "/v1/swarm/completions", payload)
    print(f"Status Code: {result['status_code']}")
    print(f"Response: {json.dumps(result['data'], indent=2)}")
    return result['success']

def test_batch_completions():
    """Test batch completions endpoint"""
    print("\n=== Batch Completions Test ===")
    
    # Test payload with multiple swarms
    payload = [
        {
            "name": "Swarm 1",
            "description": "First test swarm",
            "agents": [
                {
                    "agent_name": "Agent 1",
                    "description": "Test agent",
                    "system_prompt": "You are a helpful assistant.",
                    "model_name": "gpt-4",
                    "role": "assistant",
                    "max_loops": 1
                }
            ],
            "max_loops": 1,
            "task": "Say hello"
        },
        {
            "name": "Swarm 2",
            "description": "Second test swarm",
            "agents": [
                {
                    "agent_name": "Agent 2",
                    "description": "Test agent",
                    "system_prompt": "You are a helpful assistant.",
                    "model_name": "gpt-4",
                    "role": "assistant",
                    "max_loops": 1
                }
            ],
            "max_loops": 1,
            "task": "Say goodbye"
        }
    ]

    result = make_request("POST", "/v1/swarm/batch/completions", payload)
    print(f"Status Code: {result['status_code']}")
    print(f"Response: {json.dumps(result['data'], indent=2)}")
    return result['success']

def run_all_tests():
    """Run all tests with some delay between them"""
    results = {
        "health": False,
        "swarm": False,
        "batch": False
    }
    
    try:
        print("Starting API tests...")
        
        # Test health endpoint
        results["health"] = test_health()
        if not results["health"]:
            print("\nERROR: Health check failed. Stopping tests.")
            return results
        sleep(1)

        # Test single swarm
        results["swarm"] = test_run_swarm()
        sleep(1)

        # Test batch completions
        results["batch"] = test_batch_completions()

    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    
    print("\nTest Results Summary:")
    print("=====================")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.title():10} {status}") 