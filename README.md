# Swarms API 

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

Build, deploy, and orchestrate agents at scale with ease

## Features

- **Swarms API**: A powerful REST API for managing and executing multi-agent systems with ease.
- **Flexible Model Support**: Utilize various AI models, including GPT-4, BERT, and custom models tailored to your needs.
- **Diverse Swarm Architectures**: Choose from multiple swarm architectures such as Concurrent, Sequential, and Hybrid workflows to optimize task execution.
- **Dynamic Agent Configuration**: Easily configure agents with customizable parameters for different roles and tasks.
- **Test Suite**: Comprehensive testing framework for API endpoints, ensuring reliability and performance.
- **Docker Support**: Containerized deployment ready, facilitating easy scaling and management of your applications.
- **Supabase Integration**: Built-in database support for logging, API key management, and user authentication.
- **Real-time Monitoring**: Track swarm performance and execution metrics in real-time for better insights and adjustments.
- **Batch Processing**: Execute multiple swarm tasks simultaneously for enhanced efficiency and throughput.
- **Extensive Documentation**: Detailed guides and examples to help you get started quickly and effectively.

Read the docs [here](https://docs.swarms.world/en/latest/swarms_cloud/swarms_api/)


## Swarms API

The Swarms API provides endpoints for running single and batch agent swarm operations.

### API Endpoints

- `GET /health` - Health check endpoint
- `POST /v1/swarm/completions` - Run a single swarm completion
- `POST /v1/swarm/batch/completions` - Run multiple swarm completions in batch

### Authentication

All API endpoints (except health check) require an API key passed in the `x-api-key` header:

Acquire an api key from [here](https://swarms.world/platform/api-keys)

```bash
curl -H "x-api-key: your_api_key" -H "Content-Type: application/json" -X POST https://api.swarms.world/v1/swarm/completions
```


### Example Usage

Here's a basic example of running a swarm:

```python
# tools - search, code executor, create api

import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"

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
                "model_name": "groq/deepseek-r1-distill-qwen-32b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
            {
                "agent_name": "Economic Forecaster",
                "description": "Predicts economic trends",
                "system_prompt": "You are an expert in economic forecasting.",
                "model_name": "groq/deepseek-r1-distill-qwen-32b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
            {
                "agent_name": "Data Scientist",
                "description": "Performs data analysis",
                "system_prompt": "You are a data science expert.",
                "model_name": "groq/deepseek-r1-distill-qwen-32b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best etfs and index funds for ai and tech?",
        "output_type": "str",
        "return_history": True,
    }

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    # return response.json()
    output = response.json()

    return json.dumps(output, indent=4)


def get_logs():
    response = requests.get(
        f"{BASE_URL}/v1/swarm/logs", headers=headers
    )
    output = response.json()
    return json.dumps(output, indent=4)


if __name__ == "__main__":
    result = run_single_swarm()
    print("Swarm Result:")
    print(result)
```

## Test Suite

The project includes a comprehensive test suite in `test_api.py`. To run the tests:

```bash
# Set your API key in .env file first
SWARMS_API_KEY=your_api_key

# Run tests
python test_api.py
```

The test suite includes:
- Health check testing
- Single swarm completion testing
- Batch swarm completion testing

## Docker Configuration

To run the API using Docker:

```bash
# Build the image
docker build -t swarms-api .

# Run the container
docker run -p 8080:8080 \
  -e SUPABASE_URL=your_supabase_url \
  -e SUPABASE_KEY=your_supabase_key \
  swarms-api
```


# Todo

- [ ] Add tool usage to the swarm for every agent
- [ ] Add more conversation history. Add output list of dictionaries from the self.conversation to capture the agent outputs in a cleaner way than just a string.
- [ ] Add rag for input docs like pdf, csvs, and more, add pricing of rag depending on the number of tokens in the rag
