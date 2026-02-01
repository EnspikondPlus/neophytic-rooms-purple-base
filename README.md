# Neophytic Rooms Purple Agent Baseline

A purple agent creeated for the AgentBeats competition using the template for building [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) agents.

## Project Structure

```
src/
├─ server.py                  # Server setup and agent card configuration
├─ executor.py                # A2A request handling
├─ agent.py                   # Agent implementation
└─ messenger.py               # A2A messaging utilities
tests/
└─ test_agent.py              # Agent tests
Dockerfile                    # Docker configuration
pyproject.toml                # Python dependencies
.github/
└─ workflows/
   └─ test-and-publish.yml    # CI workflow
```

## Getting Started

To see how to make a purple agent for AgentBeats, see this [draft PR](https://github.com/RDI-Foundation/agent-template/pull/8).

## Running Server Locally
```bash
uv sync

uv run purple-server
```

## Running Tests Locally

```bash
uv sync --extra test

uv run pytest tests/test_agent.py -v
```

## Running with Green Agent

Set up the Neophytic Rooms Green Agent, then run the following.

```bash
# in one terminal
uv run purple-server --host 127.0.0.1 --port 8000

# in the green agent environment, separate terminal
uv run green-server --host 127.0.0.1 --port 9009

# in another terminal
python local_run/local_run.py
```

A bit of customization is available:
```bash
# run 20 tests of hard difficulty using standard systems
python local_run/local_run.py --count 20 --difficulty hard --no-generate

# generate and run 5 tests of medium difficulty
python local_run/local_run.py --count 5 --difficulty medium
```

## Running with Docker
```bash
docker build -t neophytic-purple .

docker run --rm neophytic-purple
```


## About the Baseline
This the baseline model for the Neophytic Rooms Green Agent and associated benchmark. It uses a simple BFS algorithm to attempt to solve tutorial and easy room systems. Mainly meant to demonstrate A2A and Docker conformity rather than act as a good solver.