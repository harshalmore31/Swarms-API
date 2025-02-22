import uuid
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from swarms import Agent, SwarmRouter

load_dotenv()


def generate_id():
    return str(uuid.uuid4())


unique_id = generate_id()


class AgentSpec(BaseModel):
    agent_name: Optional[str] = Field(
        None, description="Agent Name", max_length=100
    )
    description: Optional[str] = Field(
        None, description="Description", max_length=500
    )
    system_prompt: Optional[str] = Field(
        None, description="System Prompt", max_length=500
    )
    model_name: Optional[str] = Field(
        None, description="Model Name", max_length=500
    )
    auto_generate_prompt: Optional[bool] = Field(
        None, description="Auto Generate Prompt"
    )
    max_tokens: Optional[int] = Field(None, description="Max Tokens")
    temperature: Optional[float] = Field(
        None, description="Temperature"
    )
    role: Optional[str] = Field(None, description="Role")
    max_loops: Optional[int] = Field(None, description="Max Loops")


class Agents(BaseModel):
    name: Optional[str] = Field(
        None, description="Agent Name", max_length=100
    )
    description: Optional[str] = Field(
        None, description="Description", max_length=500
    )
    agents: Optional[List[AgentSpec]] = Field(
        None, description="Agents"
    )


class SwarmSpec(BaseModel):
    name: Optional[str] = Field(
        None, description="Swarm Name", max_length=100
    )
    description: Optional[str] = Field(
        None, description="Description", max_length=500
    )
    agents: Optional[List[AgentSpec]] = Field(
        None, description="Agents"
    )
    max_loops: Optional[int] = Field(None, description="Max Loops")
    swarm_type: Optional[str] = Field(None, description="Swarm Type")
    flow: Optional[str] = Field(None, description="Flow")
    task: Optional[str] = Field(None, description="Task")
    img: Optional[str] = Field(None, description="Img")


def create_swarm(swarm_spec: SwarmSpec) -> SwarmRouter:
    try:
        # Create agents from the swarm specification
        agents = [
            Agent(
                agent_name=agent_spec.agent_name,
                description=agent_spec.description,
                system_prompt=agent_spec.system_prompt,
                model_name=agent_spec.model_name,
                auto_generate_prompt=agent_spec.auto_generate_prompt,
                max_tokens=agent_spec.max_tokens,
                temperature=agent_spec.temperature,
                role=agent_spec.role,
                max_loops=agent_spec.max_loops,
            )
            for agent_spec in swarm_spec.agents
        ]

        print(agents)

        # Create and configure the swarm
        swarm = SwarmRouter(
            name=swarm_spec.name,
            description=swarm_spec.description,
            agents=agents,
            max_loops=swarm_spec.max_loops,
            swarm_type=swarm_spec.swarm_type,
            output_type="all",
        )

        task = swarm_spec.task
        print(task)

        # Run the swarm task
        output = swarm.run(task=swarm_spec.task)
        print(output)

        return output
    except Exception as e:
        logger.error("Error creating swarm: {}", str(e))
        raise e


example_swarm = {
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

swarm_spec = SwarmSpec(**example_swarm)
output = create_swarm(swarm_spec)
print(output)
