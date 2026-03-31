#!/usr/bin/env python3
"""gRPC server entry point for the Scheduling Engine.

Usage:
    python grpc_server.py                  # default port 50051
    python grpc_server.py --port 50052     # custom port
    SCH_GRPC_PORT=50051 python grpc_server.py

The server exposes:
    - RunSchedule   — run the Adaptive GA and stream progress
    - ValidateData  — validate input data without running the solver
    - Ping          — liveness check
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.proto import scheduler_pb2_grpc  # noqa: E402
from src.grpc_service import SchedulerEngineServicer  # noqa: E402

logger = logging.getLogger("scheduler_grpc")


def serve(port: int = 50051, max_workers: int = 4) -> None:
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    scheduler_pb2_grpc.add_SchedulerEngineServicer_to_server(
        SchedulerEngineServicer(), server
    )
    address = f"[::]:{port}"
    server.add_insecure_port(address)
    server.start()
    logger.info("Scheduler Engine gRPC server listening on %s", address)
    print(f"Scheduler Engine gRPC server listening on {address}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Scheduling Engine gRPC server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SCH_GRPC_PORT", "50051")),
        help="Port to listen on (env: SCH_GRPC_PORT, default: 50051)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SCH_GRPC_WORKERS", "4")),
        help="Max worker threads (default: 4)",
    )
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)


if __name__ == "__main__":
    main()
