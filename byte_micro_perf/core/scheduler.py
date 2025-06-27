import os
import sys
import json
import time
import signal
import pathlib
import traceback
import prettytable

from typing import List, Dict, Any
import torch.distributed as dist
import torch.multiprocessing as mp

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.creators import get_backend_cls, get_op_cls
from core.creators import create_backend, create_op

class Scheduler:
    def __init__(self, args):
        # sub process control
        self._subprocesses = []
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, exiting...")
            self.__clean_subprocess()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # store args
        self.args_device = args.device
        self.backend_type = args.hardware_type
        self.disable_parallel = args.disable_parallel
        self.profiling = not args.disable_profiling
        self.report_dir = args.report_dir


        # create backend instance
        self.backend = create_backend(args.hardware_type)

        self.backend.node_world_size = args.node_world_size
        self.backend.node_rank = args.node_rank

        self.backend.numa_world_size = args.numa_world_size
        self.backend.numa_rank = args.numa_rank

        self.backend.all_process_size = args.all_process_size
        self.backend.all_process_rank = args.all_process_rank



    def prepare_task(self, task):
        # get op cls
        self.op_name = task
        self.op_cls = get_op_cls(self.backend_type, self.op_name)
        self.is_concurrent = self.op_cls.is_concurrent()

        # get device info
        if self.args_device == "all":
            ori_target_devices = self.backend.avail_devices
        else:
            try:
                ori_target_devices = []
                for d in self.args_device.split(","):
                    if d.isdigit() and int(d) < self.backend.device_count:
                        ori_target_devices.append(int(d))
            except Exception as e:
                logger.error(f"invalid device config: {self.args_device}, error msg: {e}")
                return
        ori_target_device_count = len(ori_target_devices)
        if ori_target_device_count == 0:
            logger.error("no valid device")
            return


        # for each node, each numa process, provide:
        # 1. ori_target_devices: [0, 1, 2, 3, 4, 5, 6, 7]
        # 2. total_target_devices: [0, 1, 2 ,3, 4, 5, 6, 7]
        # 3. target_devices: [0, 1, 2, 3] or [4, 5, 6, 7]
        # 4. device_num_per_numa: 4
        # 5. all_node_devices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7] for 2 nodes
        # one node, one process, specified devices
        if not self.is_concurrent:
            if self.backend.numa_rank == 0:
                self.backend.ori_target_devices = ori_target_devices
                self.backend.target_devices = ori_target_devices if not self.disable_parallel else ori_target_devices[0:1]
                self.backend.total_target_devices = ori_target_devices if not self.disable_parallel else ori_target_devices[0:1]
                self.backend.device_num_per_numa = ori_target_device_count if not self.disable_parallel else 1
                self.backend.all_node_devices = ori_target_devices
            else:
                return None

        # multiple nodes, multiple processes, specified devices on each
        else:
            total_target_devices = []
            target_devices = []
            device_num_per_numa = ori_target_device_count // self.backend.numa_world_size

            if device_num_per_numa == 0:
                if self.backend.numa_rank < ori_target_device_count:
                    target_devices = ori_target_devices[self.backend.numa_rank:self.backend.numa_rank+1]
                total_target_devices = ori_target_devices
            else:
                target_devices = ori_target_devices[self.backend.numa_rank * device_num_per_numa:(self.backend.numa_rank + 1) * device_num_per_numa]
                total_target_devices = ori_target_devices[0:self.backend.numa_world_size * device_num_per_numa]

            temp_all_node_devices = [None for _ in range(self.backend.all_process_size)]
            dist.all_gather_object(temp_all_node_devices, target_devices)

            if len(target_devices) == 0:
                return

            all_node_devices = []
            for process_target_devices in temp_all_node_devices:
                all_node_devices.extend(process_target_devices)



            self.backend.ori_target_devices = ori_target_devices
            self.backend.total_target_devices = total_target_devices
            self.backend.target_devices = target_devices
            self.backend.device_num_per_numa = device_num_per_numa
            self.backend.all_node_devices = all_node_devices
            



        # print readable config
        pt = prettytable.PrettyTable()
        pt.align = "l"
        pt.field_names = ["config", "value"]
        pt.add_row(["backend", self.backend_type])
        pt.add_row(["op_name", self.op_name])
        pt.add_row(["is_concurrent", self.is_concurrent])

        pt.add_row(["node_rank", f"{self.backend.node_rank} / {self.backend.node_world_size}"])
        pt.add_row(["numa_rank", f"{self.backend.numa_rank} / {self.backend.numa_world_size}"])
        pt.add_row(["process_rank", f"{self.backend.all_process_rank} / {self.backend.all_process_size}"])
        pt.add_row(["avail_devices", self.backend.avail_devices])
        pt.add_row(["ori_target_devices", self.backend.ori_target_devices])
        pt.add_row(["total_target_devices", self.backend.total_target_devices])
        pt.add_row(["target_devices", self.backend.target_devices])
        pt.add_row(["all_node_devices", self.backend.all_node_devices])
        pt.add_row(["device_num_per_numa", self.backend.device_num_per_numa])
        print(pt)

        return True


    def __del__(self):
        self.__clean_subprocess()


    def __clean_subprocess(self):
        for process in self._subprocesses:
            if process.is_alive():
                pid = process.pid
                if pid is not None:
                    os.kill(pid, signal.SIGTERM)
        self._subprocesses.clear()



    def run(self, test_cases):
        self.__clean_subprocess()

        instance_num = len(self.backend.target_devices)
        input_queues = mp.Queue()
        output_queues = mp.Queue()
        try:
            _subprocess = mp.spawn(
                fn=self.subprocess_func,
                args=(input_queues, output_queues),
                nprocs=instance_num,
                join=False,
                daemon=False
            )
        except Exception as e:
            logger.error(f"Create subprocesses failed, error msg: {e}")
            return []

        self._subprocesses = _subprocess.processes
        for _ in range(instance_num):
            assert "ready" == output_queues.get()
        logger.info("all ranks are ready and listening, init done")

        result_list = []
        if self.backend.numa_rank == 0:
            valid_case_set = set()
            for index, test_case in enumerate(test_cases):
                world_size = test_case.get("world_size", 1)
                if world_size <= len(self.backend.all_node_devices):
                    test_case["index"] = index
                    valid_case_set.add(index)
                    input_queues.put(test_case, False)

            if not self.is_concurrent:      
                for _ in range(instance_num):
                    input_queues.put(None, False)
            else:
                input_queues.put(None, False)
        
            while len(valid_case_set) > 0:
                result_json = output_queues.get()
                result_list.append(result_json)
                valid_case_set.remove(result_json["arguments"]["index"])

            result_list = sorted(result_list, key=lambda x: x["arguments"]["index"])
            for result in result_list:
                result["sku_name"] = self.backend.get_device_name()
                result["op_name"] = self.op_name
                result["arguments"].pop("index")

        for process in self._subprocesses:
            process.join()

        return result_list



    def subprocess_func(self, instance_rank : int, *args): 
        try:
            input_queues, output_queues = args
            backend = self.backend

            # computation ops
            if not self.is_concurrent:
                # world_size: 8
                # index: [0, 1, 2, 3, 4, 5, 6, 7]
                # device_id: [0, 1, 2, 3, 4, 5, 6, 7]
                true_world_size = len(backend.target_devices)
                true_rank = backend.numa_rank * backend.device_num_per_numa + instance_rank
                true_device_index = backend.target_devices[true_rank]
                print(f"true_world_size: {true_world_size}, true_rank: {true_rank}, true_device_index: {true_device_index}")
                backend.set_device(true_device_index)
                backend.true_device_index = true_device_index

                # device process is ready
                output_queues.put("ready")

                # loop function
                while True:
                    # get task args
                    test_case = input_queues.get()
                    if test_case is None:
                        break
                
                    # try create op
                    op_instance = None
                    try:
                        op_instance = create_op(
                            self.op_name, test_case, backend
                        )
                    except Exception as e:
                        print(traceback.format_exc())

                    # try bench op
                    result_json = {
                        "arguments": test_case, 
                        "targets": {}    
                    }
                    if op_instance is not None:
                        latency_us = 0.
                        kernel_list = []
                        try:
                            latency_us, kernel_list = backend.perf(
                                op_instance, 
                                profiling=self.profiling
                            )
                        except Exception as e:
                            print(traceback.format_exc())
                        result_json["provider"] = op_instance.get_provider()
                        result_json["targets"] = op_instance.summary(latency_us)
                        result_json["kernels"] = kernel_list
                    arguments_str = json.dumps(result_json["arguments"])
                    targets_str = json.dumps(result_json["targets"], indent=4)
                    print(f"{arguments_str}\n{targets_str}\n")         
                    output_queues.put(result_json, block=False)
            

            # communication ops
            else:
                # world_size: 16
                # index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # device_id: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
                true_world_size = len(backend.all_node_devices)
                true_rank = backend.all_process_rank * backend.device_num_per_numa + instance_rank
                true_device_index = backend.all_node_devices[true_rank]
                print(f"true_world_size: {true_world_size}, true_rank: {true_rank}, true_device_index: {true_device_index}")
                backend.set_device(true_device_index)
                backend.true_device_index = true_device_index

                # init dist env
                dist_module = backend.get_dist_module()
                backend.initialize_ccl(true_rank, true_world_size)                

                # device process is ready
                output_queues.put("ready")

                # store process group for each group_size config
                process_groups_mapping = {true_world_size: None}

                # loop function
                while True:
                    # sync on all devices, get task args
                    broadcast_test_case = [None]
                    if true_rank == 0:
                        test_case = input_queues.get()
                        broadcast_test_case[0] = test_case
                    if true_world_size > 1:
                        dist_module.broadcast_object_list(broadcast_test_case, 0)
                    test_case = broadcast_test_case[0]
                    if test_case is None:
                        break
                    
                    result_list = [None for _ in range(true_world_size)]

                    # according to given world_size, some devices work, some devices sleep
                    world_size = test_case.get("world_size", 1)
                    if world_size > 1 and world_size not in process_groups_mapping:
                        process_groups_mapping[world_size] = backend.new_group(range(world_size))

                    if true_rank < world_size:
                        # try create op
                        op_instance = None
                        try:
                            op_instance = create_op(
                                self.op_name, test_case, backend, 
                                op_group=process_groups_mapping.get(world_size, None), 
                                group_size=world_size
                            )
                        except Exception as e:
                            print(traceback.format_exc())

                        # try bench op
                        result_list[true_rank] = {
                            "arguments": test_case, 
                            "targets": {}    
                        }
                        if op_instance is not None:
                            latency_us = 0.
                            kernel_list = []
                            try:
                                latency_us, kernel_list = backend.perf(
                                    op_instance, 
                                    profiling=self.profiling
                                )
                            except Exception as e:
                                print(traceback.format_exc())
                            result_list[true_rank]["provider"] = op_instance.get_provider()
                            result_list[true_rank]["targets"] = op_instance.summary(latency_us)
                            result_list[true_rank]["kernels"] = kernel_list

                    # sync on all devices, gather all results    
                    if true_world_size > 1:
                        dist_module.all_gather_object(result_list, result_list[true_rank])
                        
                    if backend.numa_rank == 0 and instance_rank == 0:
                        if result_list[0]["targets"]:
                            latency_list = [result["targets"]["latency(us)"] for result in result_list[:world_size]]
                            algo_bw_list = [result["targets"]["algo_bw(GB/s)"] for result in result_list[:world_size]]
                            bus_bw_list = [result["targets"]["bus_bw(GB/s)"] for result in result_list[:world_size]]

                            result_list[0]["targets"]["latency(us)"] = min(latency_list)
                            result_list[0]["targets"]["algo_bw(GB/s)"] = max(algo_bw_list)
                            result_list[0]["targets"]["bus_bw(GB/s)"] = max(bus_bw_list)
                            result_list[0]["targets"]["algo_bw_sum(GB/s)"] = round(sum(algo_bw_list), 3)
                            result_list[0]["targets"]["bus_bw_sum(GB/s)"] = round(sum(bus_bw_list), 3)
                            result_list[0]["targets"]["latency_list(us)"] = latency_list
                            result_list[0]["targets"]["algo_bw_list(GB/s)"] = algo_bw_list
                            result_list[0]["targets"]["bus_bw_list(GB/s)"] = bus_bw_list

                        arguments_str = json.dumps(result_list[0]["arguments"])
                        targets_str = json.dumps(result_list[0]["targets"], indent=4)
                        print(f"device {backend.total_target_devices[:world_size]}\n{arguments_str}\n{targets_str}\n")

                        output_queues.put(result_list[0], block=False)

                backend.destroy_process_group()


        except Exception as e:
            traceback.print_exc()
            sys.exit(1)
