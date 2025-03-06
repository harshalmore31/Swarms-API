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
- **Speech Processing**: Text-to-speech and speech-to-text capabilities for voice-enabled applications
- **Document Analysis**: Built-in RAG (Retrieval Augmented Generation) support for processing PDFs, CSVs, and other documents
- **Streaming Support**: Real-time streaming of swarm outputs for responsive applications
- **Usage Tracking**: Monitor API usage and credit consumption

Read the docs [here](https://docs.swarms.world/en/latest/swarms_cloud/swarms_api/)


## Swarms API

The Swarms API provides endpoints for running single and batch agent swarm operations.

### API Endpoints & Parameters

#### Core Swarm Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/v1/swarm/completions` | POST | Run a single swarm completion | See Swarm Parameters table below |
| `/v1/swarm/batch/completions` | POST | Run multiple swarm completions | Array of swarm parameters |
| `/v1/swarm/stream/completions` | POST | Stream swarm completion results | Same as completions + `stream: true` |
| `/v1/swarm/stream/batch` | POST | Stream batch completion results | Array of swarm parameters + `stream: true` |
| `/v1/swarm/logs` | GET | Retrieve swarm execution logs | `limit`, `offset`, `start_date`, `end_date` |

#### Speech & Audio Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/v1/speech/text-to-speech` | POST | Convert text to speech | `text`, `voice_id`, `output_format` |
| `/v1/speech/speech-to-text` | POST | Convert speech to text | `audio_file`, `language`, `model` |
| `/v1/speech/transcribe` | POST | Transcribe audio files | `audio_file`, `language`, `timestamps` |

#### Document Processing Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/v1/docs/process` | POST | Process documents | `file`, `type`, `chunk_size`, `overlap` |
| `/v1/docs/query` | POST | Query processed docs | `query`, `doc_id`, `k_results` |
| `/v1/docs/status` | GET | Check processing status | `doc_id` |

#### Management Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/v1/usage` | GET | Get API usage stats | `start_date`, `end_date` |
| `/v1/credits` | GET | Check API credits | None |
| `/v1/models` | GET | List available models | `type`, `provider` |

### Swarm Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Name of the swarm |
| `description` | string | No | Description of the swarm's purpose |
| `agents` | array | Yes | Array of agent configurations |
| `max_loops` | integer | No | Maximum iteration loops (default: 1) |
| `swarm_type` | string | Yes | Type of workflow ("ConcurrentWorkflow", "SequentialWorkflow", "HybridWorkflow") |
| `task` | string | Yes | The task to be performed |
| `output_type` | string | No | Desired output format (default: "str") |
| `return_history` | boolean | No | Include conversation history (default: false) |
| `stream` | boolean | No | Enable streaming response (default: false) |

### Agent Configuration Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_name` | string | Yes | Name of the agent |
| `description` | string | No | Agent's purpose description |
| `system_prompt` | string | Yes | System prompt for the agent |
| `model_name` | string | Yes | AI model to use |
| `role` | string | Yes | Agent's role ("worker", "manager", etc.) |
| `max_loops` | integer | No | Maximum loops for this agent |
| `max_tokens` | integer | No | Maximum tokens for responses |
| `temperature` | float | No | Response randomness (0-2) |
| `tools` | array | No | List of tools available to the agent |

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

- [ ] Add tool usage to the swarm for every agent --> Tool dictionary input for every agent -> agent outputs dictionary of that tool usage
- [x] Add more conversation history. Add output list of dictionaries from the self.conversation to capture the agent outputs in a cleaner way than just a string.
- [x] Add rag for input docs like pdf, csvs, and more, add pricing of rag depending on the number of tokens in the rag
- [x] Add async streaming output 
- [ ] Add autonomous agent builder if the user doesn't upload agents, we should make them autonomously through the agent builder
- [ ] Integrate gunicorn to make the api faster
- [x] Add speech inputs and speech outputs as well and charge more credits for that