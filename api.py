import os
import time
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import supabase
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from swarms import Agent, SwarmRouter
from swarms.utils.litellm_tokenizer import count_tokens

load_dotenv()


def generate_id():
    return str(uuid.uuid4())


unique_id = generate_id()


class AgentSpec(BaseModel):
    agent_name: Optional[str] = Field(
        None, title="Agent Name", max_length=100
    )
    description: Optional[str] = Field(
        None, title="Description", max_length=500
    )
    system_prompt: Optional[str] = Field(
        None, title="System Prompt", max_length=500
    )
    model_name: Optional[str] = Field(
        None, title="Model Name", max_length=500
    )
    auto_generate_prompt: Optional[bool] = Field(
        None, title="Auto Generate Prompt"
    )
    max_tokens: Optional[int] = Field(None, title="Max Tokens")
    temperature: Optional[float] = Field(None, title="Temperature")
    role: Optional[str] = Field(None, title="Role")
    max_loops: Optional[int] = Field(None, title="Max Loops")


class Agents(BaseModel):
    name: Optional[str] = Field(
        None, title="Agent Name", max_length=100
    )
    description: Optional[str] = Field(
        None, title="Description", max_length=500
    )
    agents: Optional[List[AgentSpec]] = Field(None, title="Agents")


class SwarmSpec(BaseModel):
    name: Optional[str] = Field(
        None, title="Swarm Name", max_length=100
    )
    description: Optional[str] = Field(
        None, title="Description", max_length=500
    )
    agents: Optional[List[AgentSpec]] = Field(None, title="Agents")
    max_loops: Optional[int] = Field(None, title="Max Loops")
    swarm_type: Optional[str] = Field(None, title="Swarm Type")
    flow: Optional[str] = Field(None, title="Flow")
    task: Optional[str] = Field(None, title="Task")
    img: Optional[str] = Field(None, title="Img")


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

        print(swarm)

        # Run the swarm task
        output = swarm.run(task=swarm_spec.task)

        print(output)
        print(type(output))
        return output
    except Exception as e:
        logger.error("Error creating swarm: {}", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create swarm: {str(e)}",
        )


def get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    return supabase.create_client(supabase_url, supabase_key)


def check_api_key(api_key: str) -> bool:
    supabase_client = get_supabase_client()
    response = (
        supabase_client.table("swarms_cloud_api_keys")
        .select("*")
        .eq("key", api_key)
        .execute()
    )
    return bool(response.data)


def get_user_id_from_api_key(api_key: str) -> str:
    """
    Maps an API key to its associated user ID.

    Args:
        api_key (str): The API key to look up

    Returns:
        str: The user ID associated with the API key

    Raises:
        ValueError: If the API key is invalid or not found
    """
    supabase_client = get_supabase_client()
    response = (
        supabase_client.table("swarms_cloud_api_keys")
        .select("user_id")
        .eq("key", api_key)
        .execute()
    )
    if not response.data:
        raise ValueError("Invalid API key")
    return response.data[0]["user_id"]


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Dependency to verify the API key.
    """
    if not check_api_key(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid API Key")


def deduct_credits(
    api_key: str, amount: float, product_name: str
) -> None:
    """
    Deducts the specified amount of credits for the user identified by api_key,
    preferring to use free_credit before using regular credit, and logs the transaction.
    """
    supabase_client = get_supabase_client()
    user_id = get_user_id_from_api_key(api_key)

    # 1. Retrieve the user's credit record
    response = (
        supabase_client.table("swarms_cloud_users_credits")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User credits record not found.",
        )

    record = response.data[0]
    # Use Decimal for precise arithmetic
    available_credit = Decimal(record["credit"])
    free_credit = Decimal(record.get("free_credit", "0"))
    deduction = Decimal(str(amount))

    print(
        f"Available credit: {available_credit}, Free credit: {free_credit}, Deduction: {deduction}"
    )

    # 2. Verify sufficient total credits are available
    if (available_credit + free_credit) < deduction:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient credits.",
        )

    # 3. Log the transaction
    log_response = (
        supabase_client.table("swarms_cloud_services")
        .insert(
            {
                "user_id": user_id,
                "api_key": api_key,
                "charge_credit": int(
                    deduction
                ),  # Assuming credits are stored as integers
                "product_name": product_name,
            }
        )
        .execute()
    )
    if not log_response.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log the credit transaction.",
        )

    # 4. Deduct credits: use free_credit first, then deduct the remainder from available_credit
    if free_credit >= deduction:
        free_credit -= deduction
    else:
        remainder = deduction - free_credit
        free_credit = Decimal("0")
        available_credit -= remainder

    update_response = (
        supabase_client.table("swarms_cloud_users_credits")
        .update(
            {
                "credit": str(available_credit),
                "free_credit": str(free_credit),
            }
        )
        .eq("user_id", user_id)
        .execute()
    )
    if not update_response.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update credits.",
        )


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Swarm Agent API",
    description="API for managing and executing Python agents in the cloud without Docker/Kubernetes.",
    version="1.0.0",
    debug=True,
)

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def calculate_swarm_cost(
    agents: List[Agent],
    input_text: str,
    execution_time: float,
    agent_outputs: Union[
        List[str], str
    ] = None,  # Add agent outputs parameter
) -> Dict[str, Any]:
    """
    Calculate the cost of running a swarm based on agents, tokens, and execution time.
    Includes system prompts, agent memory, and scaled output costs.

    Args:
        agents: List of agents used in the swarm
        input_text: The input task/prompt text
        execution_time: Time taken to execute in seconds
        agent_outputs: List of output texts from each agent

    Returns:
        Dict containing cost breakdown and total cost
    """
    # Base costs per unit (these could be moved to environment variables)
    COST_PER_AGENT = 0.01  # Base cost per agent
    COST_PER_1M_INPUT_TOKENS = 5.00  # Cost per 1M input tokens
    COST_PER_1M_OUTPUT_TOKENS = (
        15.50  # Cost per 1M output tokens (2.5x input cost)
    )

    try:
        # Calculate input tokens for task
        task_tokens = count_tokens(input_text)

        # Calculate total input tokens including system prompts and memory for each agent
        total_input_tokens = 0
        total_output_tokens = 0
        per_agent_tokens = {}

        for i, agent in enumerate(agents):
            agent_input_tokens = task_tokens  # Base task tokens

            # Add system prompt tokens if present
            if agent.system_prompt:
                agent_input_tokens += count_tokens(
                    agent.system_prompt
                )

            # Add memory tokens if available
            try:
                memory = agent.short_memory.return_history_as_string()
                if memory:
                    memory_tokens = count_tokens(str(memory))
                    agent_input_tokens += memory_tokens
            except Exception as e:
                logger.warning(
                    f"Could not get memory for agent {agent.agent_name}: {str(e)}"
                )

            # Calculate actual output tokens if available, otherwise estimate
            if agent_outputs:
                if isinstance(agent_outputs, list) and i < len(
                    agent_outputs
                ):
                    agent_output_tokens = count_tokens(
                        agent_outputs[i]
                    )
                elif isinstance(agent_outputs, str):
                    agent_output_tokens = count_tokens(agent_outputs)
                else:
                    agent_output_tokens = int(
                        agent_input_tokens * 2.5
                    )  # Estimated output tokens
            else:
                agent_output_tokens = int(
                    agent_input_tokens * 2.5
                )  # Estimated output tokens

            # Store per-agent token counts
            per_agent_tokens[agent.agent_name] = {
                "input_tokens": agent_input_tokens,
                "output_tokens": agent_output_tokens,
                "total_tokens": agent_input_tokens
                + agent_output_tokens,
            }

            # Add to totals
            total_input_tokens += agent_input_tokens
            total_output_tokens += agent_output_tokens

        # Calculate costs (convert to millions of tokens)
        agent_cost = len(agents) * COST_PER_AGENT
        input_token_cost = (
            (total_input_tokens / 1_000_000)
            * COST_PER_1M_INPUT_TOKENS
            * len(agents)
        )
        output_token_cost = (
            (total_output_tokens / 1_000_000)
            * COST_PER_1M_OUTPUT_TOKENS
            * len(agents)
        )

        # Calculate total cost
        total_cost = agent_cost + input_token_cost + output_token_cost

        output = {
            "cost_breakdown": {
                "agent_cost": round(agent_cost, 6),
                "input_token_cost": round(input_token_cost, 6),
                "output_token_cost": round(output_token_cost, 6),
                "token_counts": {
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens
                    + total_output_tokens,
                    "per_agent": per_agent_tokens,
                },
                "num_agents": len(agents),
                "execution_time_seconds": round(execution_time, 2),
            },
            "total_cost": round(total_cost, 6),
        }

        # return json.dumps(output, indent=4)
        return output

    except Exception as e:
        logger.error(f"Error calculating swarm cost: {str(e)}")
        raise ValueError(f"Failed to calculate swarm cost: {str(e)}")


# # Build the agents
# @app.post(
#     "/v1/agents",
#     dependencies=[
#         Depends(verify_api_key),
#         Depends(rate_limit_dependency),
#     ],
# )
# async def build_agents(
#     agents: Agents,
# ) -> Dict[str, Any]:
#     # Verify API key

#     # Deduct credits for agent creation
#     # deduct_credits(x_api_key, 1.0, "agent_creation")

#     try:
#         # Create list to store created agents
#         created_agents = []

#         # Create each agent and add to registry
#         for agent_spec in agents.agents:
#             agent = create_agent(agent_spec)
#             registry.add_agent(agent)
#             created_agents.append(
#                 {"name": agent.agent_name, "id": agent.id}
#             )

#         return {
#             "status": "success",
#             "message": f"Successfully created {len(created_agents)} agents",
#             "agents": created_agents,
#             "name": agents.name,
#             "description": agents.description,
#         }

#     except Exception as e:
#         logger.error(f"Error creating agents: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create agents: {str(e)}",
#         )


# @app.get(
#     "/v1/agents/{team_name}",
#     dependencies=[
#         Depends(verify_api_key),
#         Depends(rate_limit_dependency),
#     ],
# )
# async def get_agents_info() -> Dict[str, Any]:
#     # Verify API key

#     try:
#         # Get all agents from registry
#         agents = registry.return_all_agents()

#         # Compile agent statistics and info
#         agents_info = []
#         for agent in agents:
#             agents_info.append(
#                 {
#                     "name": agent.agent_name,
#                     "id": agent.id,
#                     "description": agent.description,
#                     "model_name": agent.model_name,
#                     "role": agent.role,
#                 }
#             )

#         return {
#             "status": "success",
#             "total_agents": len(agents),
#             "agents": agents_info,
#             "registry_id": registry.id,
#             "stats": {
#                 "active_agents": len(
#                     [a for a in agents if a.is_active]
#                 ),
#                 "models_used": list(
#                     set(a.model_name for a in agents if a.model_name)
#                 ),
#             },
#         }

#     except Exception as e:
#         logger.error(f"Error fetching agents info: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to fetch agents info: {str(e)}",
#         )


@app.get("/")
def root():
    return {
        "status": "Welcome to the SwarmCloud API. Check out the docs at https://docs.swarms.world"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/v1/swarm/completions",
    dependencies=[
        Depends(verify_api_key),
    ],
)
async def run_swarm(
    swarm: SwarmSpec,
    x_api_key: str = Header(...),
) -> Dict[str, Any]:
    """
    Run a swarm with the specified task.
    """
    try:
        swarm_name = swarm.name

        agents = swarm.agents

        # Log start of swarm execution
        logger.info(
            f"Starting swarm {swarm_name} with {len(agents)} agents"
        )
        start_time = time.time()

        # Create and run the swarm
        logger.debug(f"Creating swarm object for {swarm_name}")
        result = create_swarm(swarm)
        print(result)
        logger.debug(f"Running swarm task: {swarm.task}")

        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(
            f"Swarm {swarm_name} executed in {round(execution_time, 2)} seconds"
        )

        # Calculate costs
        logger.debug(f"Calculating costs for swarm {swarm_name}")
        cost_info = calculate_swarm_cost(
            agents=agents,
            input_text=swarm.task,
            agent_outputs=result,
            execution_time=execution_time,
        )
        logger.info(
            f"Cost calculation completed for swarm {swarm_name}: {cost_info}"
        )

        # Deduct credits based on calculated cost
        logger.debug(
            f"Deducting credits for swarm {swarm_name} with cost {cost_info['total_cost']}"
        )

        deduct_credits(
            x_api_key,
            cost_info["total_cost"],
            f"swarm_execution_{swarm_name}",
        )

        # Format the response
        response = {
            "status": "success",
            "swarm_name": swarm_name,
            "description": swarm.description,
            "task": swarm.task,
            "metadata": {
                "max_loops": swarm.max_loops,
                "num_agents": len(agents),
                "execution_time_seconds": round(execution_time, 2),
                "completion_time": time.time(),
                "billing_info": cost_info,
            },
            "output": result,
        }
        logger.info(response)
        return response

    except HTTPException as http_exc:
        logger.error("HTTPException occurred: {}", http_exc.detail)
        raise
    except Exception as e:
        logger.error("Error running swarm {}: {}", swarm_name, str(e))
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run swarm: {str(e)}",
        )


# --- Main Entrypoint ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
