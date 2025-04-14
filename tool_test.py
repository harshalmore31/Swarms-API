import os
import json
import requests
from dotenv import load_dotenv
import time

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "http://localhost:8080"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


def get_current_weather(location, unit="fahrenheit"):
    if location == "Boston, MA":
        return f"The weather is {12 if unit == 'fahrenheit' else -11}°{'F' if unit == 'fahrenheit' else 'C'}"
    elif location == "San Francisco, CA":
        return f"The weather is {65 if unit == 'fahrenheit' else 18}°{'F' if unit == 'fahrenheit' else 'C'}"
    return f"Weather information for {location} not found"

tools_list = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]


def run_single_swarm():
    # STEP 1: Initial request to get function call
    initial_payload = {
        "name": "Swarms with Function Calling",
        "description": "You are an AI with tools",
        "agents": [
            {
                "agent_name": "AI assistant",
                "description": "You are an helpful assistant",
                "system_prompt": "You are a helpful assistant. Always use the weather tool when asked about weather.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7,
                "auto_generate_prompt": False,
                "tools": tools_list
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "what is weather in Boston right now? Should I wear a jacket?",
        "output_type": "dict"
    }

    print("STEP 1: Sending initial swarm request...")
    response = requests.post(f"{BASE_URL}/v1/swarm/completions", headers=headers, json=initial_payload)
    print("Status Code:", response.status_code)
    
    try:
        result = response.json()
        print("\nInitial Response:")
        print(json.dumps(result, indent=4)[:1000] + "..." if len(json.dumps(result)) > 1000 else json.dumps(result, indent=4))
        
        # Extract function call details from the response
        function_args = None
        
        # First check if output is a list (handle that format)
        if "output" in result and isinstance(result["output"], list):
            # Look for the assistant's response in the list
            for message in result["output"]:
                if message.get("role") == "AI assistant" and message.get("content"):
                    try:
                        # Extract the message content removing timestamp
                        content = message.get("content")
                        if "Time:" in content and "\n" in content:
                            content = content.split("\n", 1)[1].strip()
                            
                        # Try to parse the content as JSON which might contain function arguments
                        content_json = json.loads(content)
                        if "location" in content_json:
                            function_args = content_json
                            print(f"\nExtracted function arguments from message content: {function_args}")
                            break
                    except Exception as e:
                        print(f"Failed to parse content as JSON: {e}")
                        continue
        
        # STEP 2: If we have function arguments, execute the function and send a follow-up
        if function_args and "location" in function_args:
            # Execute the function with the extracted arguments
            tool_result = get_current_weather(**function_args)
            print(f"\nFunction result: {tool_result}")
            
            # STEP 3: Send a completely new request with the result
            print("\nSTEP 2: Sending separate follow-up request...")
            followup_payload = {
                "name": "Swarms with Function Calling",
                "description": "You are an AI with tools",
                "agents": [
                    {
                        "agent_name": "AI assistant",
                        "description": "You are a helpful assistant",
                        "system_prompt": "You are a helpful assistant. When answering questions about weather, provide detailed responses that include recommendations based on the temperature.",
                        "model_name": "openai/gpt-4o",
                        "role": "worker",
                        "max_loops": 1,
                        "max_tokens": 8192,
                        "temperature": 0.7,
                        "auto_generate_prompt": False
                    },
                ],
                "max_loops": 1,
                "swarm_type": "ConcurrentWorkflow",
                "task": f"The current weather in Boston is {tool_result}. Based on this information, should I wear a jacket?",
                "output_type": "dict"
            }
            
            followup_response = requests.post(f"{BASE_URL}/v1/swarm/completions", headers=headers, json=followup_payload)
            print("Follow-up Status Code:", followup_response.status_code)
            
            # Parse the final response
            if followup_response.status_code == 200:
                final_result = followup_response.json()
                print("\nFinal Response:")
                print(json.dumps(final_result, indent=4))
                
                # Extract the assistant's final answer
                if "output" in final_result and isinstance(final_result["output"], list):
                    for message in final_result["output"]:
                        if message.get("role") == "AI assistant" and message.get("content"):
                            content = message.get("content")
                            if "Time:" in content and "\n" in content:
                                content = content.split("\n", 1)[1].strip()
                            
                            if content.strip():  # Check if content is not empty
                                print("\nFinal Answer:")
                                print(content)
                            else:
                                print("\nEmpty response from assistant in the final answer.")
                
                return final_result
        else:
            print("\nNo function arguments found in the response")
        
        return result
    except Exception as ex:
        print("Error during function calling process:", ex)
        print("Response Text:", response.text if 'response' in locals() else "No response")
        return None

def get_logs():
    response = requests.get(f"{BASE_URL}/v1/swarm/logs", headers=headers)
    try:
        return response.json()
    except Exception as ex:
        print("Error parsing logs JSON:", ex)
        return None

if __name__ == "__main__":
    # Run health check
    health = run_health_check()
    print("Health Check Response:")
    print(json.dumps(health, indent=4))
    
    # Give a short pause
    time.sleep(1)
    
    # Run a single swarm with function calling
    swarm_result = run_single_swarm()
    
    # Give logs time to be processed
    time.sleep(2)
    
    # Retrieve and print API logs
    logs = get_logs()
    print("\nLogs:")
    if logs and "logs" in logs and logs["logs"]:
        print(json.dumps(logs["logs"][-1], indent=4))  # Print the latest log entry
    else:
        print("No logs found or empty logs")