from abc import ABC, abstractmethod
from typing import Any, Dict, List
import time

class Backend(ABC):
    def __init__(self, iterations, dtype):
        self.iterations = iterations
        self.warmup = int(0.1 * iterations)
        self.execution_history = []
        self.op = None
        self.dtype = dtype

    def get_performance_data(self):
        return self.execution_history

    def gemm(self):
        pass

    def axpy(self):
        pass # 返回具体op实现

    def softmax(self):
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

        report = {
            "dtype": "float32",
            "shape": input_shapes,
            "ops(ops per-sec)" : round(self.iterations / sum(self.execution_history), 2),
            "avg latency(ms)" : round(sum(self.execution_history) * 1000 / len(self.execution_history), 2),
            "theoretical ops" : 178,
            "theoretical latency"  : 1.3,
            "theoretical io"  : 2.3,
            "mfu" : 0.87
         }
        return report
