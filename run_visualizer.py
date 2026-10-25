#!/usr/bin/env python3
"""
Quickstart launcher for the Organism Cognitive Web Visualizer.
Usage:
    python3 run_visualizer.py
    python3 run_visualizer.py --port 8080 --seed 42
"""
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualizer.server import run_server

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Organism Cognitive Web Visualizer")
    ap.add_argument("--host", default="127.0.0.1", help="host address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080, help="preferred port (default: 8080)")
    ap.add_argument("--seed", type=int, default=42, help="environment seed (default: 42)")
    args = ap.parse_args()

    from visualizer.server import run_server
    run_server(host=args.host, port=args.port, seed=args.seed)
