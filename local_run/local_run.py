#!/usr/bin/env python3

import argparse
import asyncio
import json
import re
import sys
import time
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, TextPart, Role

DEFAULT_GREEN_PORT  = 9009
DEFAULT_PURPLE_PORT = 8000

class TeeLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, "w", encoding="utf-8") if filepath else None

    def log(self, msg: str):
        print(msg)
        if self.file:
            self.file.write(msg + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            print(f"📝 Logs saved to {self.filepath}")

def smart_json_dumps(data) -> str:
    text = json.dumps(data, indent=2)
    return re.sub(
        r'\[\s*([^\[\{\]\}]+?)\s*\]',
        lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']',
        text,
        flags=re.DOTALL
    )

async def send_benchmark_request(
    green_url: str,
    purple_url: str,
    config: dict,
    logger: TeeLogger
) -> None:
    
    request_payload = {
        "participants": {"solver": purple_url},
        "config": config,
    }

    logger.log(f"\n📨 Sending benchmark request to green agent...")
    logger.log(f"   Config: {smart_json_dumps(config)}\n")

    try:
        async with httpx.AsyncClient(timeout=120) as http:
            resolver = A2ACardResolver(httpx_client=http, base_url=green_url)
            card = await resolver.get_agent_card()

            client_config = ClientConfig(httpx_client=http, streaming=True)
            client = ClientFactory(client_config).create(card)

            message = Message(
                kind="message",
                role=Role.user,
                parts=[Part(root=TextPart(kind="text", text=json.dumps(request_payload)))],
                message_id=uuid4().hex,
            )

            async for event in client.send_message(message):
                task, update = event

                if task.status and task.status.message:
                    for part in task.status.message.parts:
                        if hasattr(part.root, "text"):
                            logger.log(f"  [{task.status.state.value:>9}] {part.root.text}")

                if hasattr(update, "artifact") and update.artifact:
                    artifact = update.artifact
                    logger.log(f"\n{'=' * 70}")
                    logger.log(f"📋 Artifact: {artifact.name}")
                    logger.log(f"{'=' * 70}")
                    for part in artifact.parts:
                        if hasattr(part.root, "text"):
                            logger.log(part.root.text)
                        elif hasattr(part.root, "data"):
                            # Use our smart formatter here
                            formatted_json = smart_json_dumps(part.root.data)
                            logger.log(formatted_json)
                    logger.log(f"{'=' * 70}\n")

                if task.status.state.value in ("completed", "failed", "rejected", "canceled"):
                    logger.log(f"\n✨ Benchmark finished with state: {task.status.state.value}")
                    # Small sleep to allow the socket to close gracefully and avoid 'aclose' errors
                    await asyncio.sleep(0.1)
                    break

    except httpx.ConnectError:
        logger.log(f"❌ Error: Could not connect to Green Agent at {green_url}.")
        logger.log("   Please ensure the servers are running manually (e.g. 'uv run green-server').")
        sys.exit(1)
    except Exception as e:
        logger.log(f"❌ An error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run a Rooms benchmark task against already running agents."
    )
    
    parser.add_argument("--green-port",  type=int, default=DEFAULT_GREEN_PORT,
                        help=f"Green agent port (default {DEFAULT_GREEN_PORT})")
    parser.add_argument("--purple-port", type=int, default=DEFAULT_PURPLE_PORT,
                        help=f"Purple agent port (default {DEFAULT_PURPLE_PORT})")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host address for both servers (default 127.0.0.1)")
    parser.add_argument("--log-file", default="benchmark.log",
                        help="File to save logs to (default: benchmark.log)")
    
    parser.add_argument("--count", type=int, default=3,
                        help="Number of benchmark runs (default 3)")
    parser.add_argument("--difficulty", default="tutorial",
                        choices=["tutorial", "easy", "medium", "hard", "very_hard", "random"],
                        help="Difficulty level (default tutorial)")
    parser.add_argument("--no-generate", action="store_true", default=False,
                        help="Load runs from standard_systems.json instead of generating")
    
    args = parser.parse_args()

    green_url  = f"http://{args.host}:{args.green_port}"
    purple_url = f"http://{args.host}:{args.purple_port}"

    config = {
        "generate":   not args.no_generate,
        "count":      args.count,
        "difficulty": args.difficulty,
    }

    logger = TeeLogger(args.log_file)
    
    try:
        asyncio.run(send_benchmark_request(green_url, purple_url, config, logger))
    finally:
        logger.close()


if __name__ == "__main__":
    main()