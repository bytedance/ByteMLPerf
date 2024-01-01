import argparse
import os
import sys

from llm_perf.server.serve import serve
from llm_perf.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, required=True)
    parser.add_argument("--hardware_type", "-hw", type=str, required=True)
    parser.add_argument("--port", "-p", type=str, required=True)
    parser.add_argument("--max_batch_size", type=int, required=True)
    args = parser.parse_args()

    loglevel = os.environ.get("LOG_LEVEL", "debug")
    setup_logger(loglevel)

    serve(args)


if __name__ == "__main__":
    main()
