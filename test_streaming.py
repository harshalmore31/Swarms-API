import os
import requests
from typing import List, Optional
import time
from pydantic import BaseModel, Field
from swarms.structs.swarm_router import SwarmType
from dotenv import load_dotenv

load_dotenv()


class AgentSpec(BaseModel):
    agent_name: Optional[str] = Field(None, description="Agent Name", max_length=100)
    description: Optional[str] = Field(None, description="Description", max_length=500)
    system_prompt: Optional[str] = Field(
        None, description="System Prompt", max_length=500
    )
    model_name: Optional[str] = Field(
        "gpt-4o", description="Model Name", max_length=500
    )
    auto_generate_prompt: Optional[bool] = Field(
        False, description="Auto Generate Prompt"
    )
    max_tokens: Optional[int] = Field(None, description="Max Tokens")
    temperature: Optional[float] = Field(0.5, description="Temperature")
    role: Optional[str] = Field("worker", description="Role")
    max_loops: Optional[int] = Field(1, description="Max Loops")


class SwarmSpec(BaseModel):
    name: Optional[str] = Field(None, description="Swarm Name", max_length=100)
    description: Optional[str] = Field(None, description="Description", max_length=500)
    agents: Optional[List[AgentSpec]] = Field(None, description="Agents")
    max_loops: Optional[int] = Field(None, description="Max Loops")
    swarm_type: Optional[SwarmType] = Field(None, description="Swarm Type")
    rearrange_flow: Optional[str] = Field(None, description="Flow")
    task: Optional[str] = Field(None, description="Task")
    img: Optional[str] = Field(None, description="Img")
    return_history: Optional[bool] = Field(True, description="Return History")
    rules: Optional[str] = Field(None, description="Rules")


# Set your API key and endpoint here
API_KEY = os.getenv("SWARMS_API_KEY")  # Replace with your actual API key
BASE_URL = "http://localhost:8080"  # Replace with your actual API URL
TASK = "Explain quantum computing in simple terms."

# Create a sample swarm specification
swarm_spec = SwarmSpec(
    name="Test Streaming Swarm",
    description="A test swarm for streaming API",
    agents=[
        AgentSpec(
            agent_name="Explainer",
            description="Explains complex topics simply",
            system_prompt="You are an expert at explaining complex topics in simple terms.",
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=500,
            role="worker",
        ),
        AgentSpec(
            agent_name="Reviewer",
            description="Reviews and improves explanations",
            system_prompt="You review explanations and improve them for clarity and accuracy.",
            model_name="gpt-4o",
            temperature=0.5,
            max_tokens=500,
            role="worker",
        ),
    ],
    swarm_type="SequentialWorkflow",
    task=TASK,
    max_loops=1,
    return_history=True,
)

# Set up the request
url = f"{BASE_URL}/v1/swarm/completions/stream"
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

print(f"Testing streaming endpoint at {url}")
print(f"Task: {TASK}")
print("Sending request...")

# Make the request with stream=True to get the response as it comes
start_time = time.time()
response = requests.post(
    url=url, headers=headers, json=swarm_spec.dict(exclude_none=True), stream=True
)

# Process the streaming response
for line in response.iter_lines():
    if line:
        print(line)
