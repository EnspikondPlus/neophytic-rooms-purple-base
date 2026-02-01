import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from .executor import Executor

def main():
    parser = argparse.ArgumentParser(description="Run the purple solver agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server.")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card.")
    args = parser.parse_args()

    skill = AgentSkill(
        id="neophytic_rooms_purple_baseline",
        name="Neophytic Rooms Purple Baseline",
        description="Baseline agent for solving room system navigation challenges.",
        tags=["solver", "navigation", "planning"],
        examples=["Solve the rooms puzzle"]
    )

    agent_card = AgentCard(
        name="Neophytic Rooms Purple Baseline",
        description="A baseline purple agent that attempts to solve the Neophytic Rooms navigation puzzles",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"🟣 Purple Agent running on {args.host}:{args.port}")
    uvicorn.run(server.build(), host=args.host, port=args.port)

if __name__ == '__main__':
    main()