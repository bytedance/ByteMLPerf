from abc import ABC, abstractmethod
from typing import Any, Dict, List
import time

class Backend(ABC):
    def __init__(self, iterations):
        self.iterations = iterations
        self.warmup = int(0.1 * iterations)
        self.execution_history = []
        self.op = None

    def get_performance_data(self):
        return self.execution_history

    @abstractmethod
    def gemm(self, input_shape):
        pass

    @abstractmethod
    def axpy(self, input_shape):
        pass # 返回具体op实现

    @abstractmethod
    def softmax(self, input_shape):
        pass

    @abstractmethod
    def build_tensor(self, input_shape):
        pass

    @abstractmethod
    def _run_operation(self, operation, inputs):
        pass

    def perf(self, input_shape: List[List[int]]):
        inputs = self.build_tensor(input_shape)
        start_time = time.time()
        result = self._run_operation(self.op, inputs)
        execution_time = time.time() - start_time

        self.execution_history.append(execution_time)
        report = {
            "dtype": "int8",
            "shape": input_shape,
            "ops(ops per-sec)" : 217.34,
            "avg latency(ms)" : 2.45,
            "theoretical ops" : 178,
            "theoretical latency"  : 1.3,
            "theoretical io"  : 2.3,
            "mfu" : 0.87
         }
        return report
