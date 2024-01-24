#!/usr/bin/python3

import os

import sys
import socket
import random
import argparse
import subprocess
import logging
import json

BYTE_MLPERF_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("LANUCH")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="",
        help="The task going to be evaluted, refs to workloads/")
    parser.add_argument(
        "--hardware_type",
        default="CPU",
        help="The backend going to be evaluted, refs to backends/")
    parser.add_argument("--compile_only",
                        action='store_true',
                        help="Task will stoped after compilation finished")
    parser.add_argument("--show_task_list",
                        action='store_true',
                        help="Print all task names")
    parser.add_argument("--show_hardware_list",
                        action='store_true',
                        help="Print all hardware bytemlperf supported")
    args = parser.parse_args()

    if args.show_task_list:
        log.info("******************* Supported Task *******************")
        for file in os.listdir('workloads'):
            print(file[:-5])
    if args.show_hardware_list:
        log.info("***************** Supported Hardware Backend *****************")
        for file in os.listdir('backends'):
            if not file.endswith('.py') and not file.startswith('_'):
                print(file)
    if args.task:
        log.info("******************* Pip Package Installing *******************")
        subprocess.call([
            'python3', '-m', 'pip', 'install', 'pip', '--upgrade', '--quiet'])

        # subprocess.call([
        #     'python3', '-m', 'pip', 'install', '-r', 'requirements.txt', '--quiet'])


        cmd = f'python3 core/perf_engine.py --hardware_type {args.hardware_type} --task {args.task}'
        exit_code = subprocess.call(cmd, shell=True)
        sys.exit(exit_code)
