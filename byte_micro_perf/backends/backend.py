from abc import ABC, abstractmethod
from typing import Any, Dict, List
import time
import os
from backends.utils import write_communication_ops_report, write_computation_ops_report

class Backend(ABC):
    def __init__(self, workload_dict: Dict[str, Any]):
        self.op_name = workload_dict['operator']
        self.iterations = workload_dict['iterations']
        self.warmup = int(0.1 * workload_dict['iterations'])
        self.execution_history = []
        self.op = None
        self.dtype = workload_dict['dtype']
        # communication params
        self.rank = None
        self.world_size = None
        self.group = None
   
    def get_performance_data(self):
        return self.execution_history

    def initialize_ccl(self):
        pass

    def setup_2d_group(self):
        pass    

    def gemm(self):
        pass

    def axpy(self):
        pass # 返回具体op实现

    def softmax(self):
        pass     

    def allreduce(self):
        pass       

    @abstractmethod
    def build_tensor(self, input_shapes: List[List[int]], dtype):
        pass

    @abstractmethod
    def _run_operation(self, operation, inputs):
        pass

    def perf(self, input_shapes: List[List[int]]):
        # warmup
        for _ in range(20):
            inputs = self.build_tensor(input_shapes, self.dtype)
            self._run_operation(self.op, inputs)
    
        for _ in range(self.iterations):
            inputs = self.build_tensor(input_shapes, self.dtype)
            start_time = time.time()
            result = self._run_operation(self.op, inputs)
            execution_time = time.time() - start_time

            self.execution_history.append(execution_time)
        
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0 and self.op_name in ["allreduce"]:
            latency = round(sum(self.execution_history) * 1000 / len(self.execution_history), 2)
            report = write_communication_ops_report(self.op_name, self.dtype, input_shapes, self.group.size(), latency)
        else:
            report = write_computation_ops_report(self.dtype, input_shapes, self.iterations, self.execution_history)    
        return report 
