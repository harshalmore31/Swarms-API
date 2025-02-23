# Swarms API 

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

A radically simple, reliable, and high performance template to enable you to quickly get set up building multi-agent applications

## Features

- **Swarms API**: A powerful REST API for managing and executing multi-agent systems
- **Test Suite**: Comprehensive testing framework for API endpoints
- **Docker Support**: Containerized deployment ready
- **Supabase Integration**: Built-in database support for logging and API key management


## Swarms API

The Swarms API provides endpoints for running single and batch agent swarm operations.

### API Endpoints

- `GET /health` - Health check endpoint
- `POST /v1/swarm/completions` - Run a single swarm completion
- `POST /v1/swarm/batch/completions` - Run multiple swarm completions in batch

### Authentication

All API endpoints (except health check) require an API key passed in the `x-api-key` header:

```bash
curl -H "x-api-key: your_api_key" -H "Content-Type: application/json" -X POST https://api.swarms.world/v1/swarm/completions
```

### Example Usage

Here's a basic example of running a swarm:

```python
import requests

API_KEY = "your_api_key"
BASE_URL = "https://api.swarms.world"

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
            "max_loops": 1
        }
    ],
    "max_loops": 1,
    "swarm_type": "ConcurrentWorkflow",
    "task": "Write a short blog post about AI agents."
}

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(
    f"{BASE_URL}/v1/swarm/completions",
    headers=headers,
    json=payload
)
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

### Environment Variables

Required environment variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key
- `SWARMS_API_KEY`: For testing purposes
