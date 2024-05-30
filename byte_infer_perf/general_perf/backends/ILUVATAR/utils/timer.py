import tvm
import time
from contextlib import contextmanager


_get_timer = tvm.get_global_func("profiling.get_timer")
_start = tvm.get_global_func("profiling.start")
_stop = tvm.get_global_func("profiling.stop")
_elapse_time = tvm.get_global_func("profiling.elapse_time")


class Timer:
    def __init__(self, device=None):
        self.last_duration = 0  # ms
        self.duration_list = []  # ms
        
        self.device = device
        self._timer = None
        if device is not None:
            self._timer =  _get_timer(device)

        self.start_cnt = 0
        self.end_cnt = 0

    def total_duration(self):
        return sum(self.duration_list)

    def _update(self, duration):
        self.last_duration = duration
        self.duration_list.append(self.last_duration)


    def start(self):
        assert self._timer is not None
        self.start_cnt += 1
        self.device.sync()
        _start(self._timer)
         
    
    def stop(self):
        assert self._timer is not None
        self.end_cnt += 1
        assert self.end_cnt == self.start_cnt
    
        _stop(self._timer)
        self._update(_elapse_time(self._timer) / 1e6)  ## ns / 1e6 -> ms



    # @contextmanager
    # def timeit_sync(self, device, use_host_time=False):
    #     # NOTE(chen.chen)
    #     # not works as expected when use device timer
    #     # it seems python contextmanager always use host time?
    #     if use_host_time:
    #         device.sync()
    #         t1 = time.time()

    #         yield

    #         device.sync()
    #         t2 = time.time()
    #         self._update((t2 - t1) * 1e3)  ## s * 1e3 -> ms
    #     else:
    #         timer = _get_timer(device)
    #         device.sync()
    #         _start(timer)

    #         yield

    #         _stop(timer)
    #         self._update(_elapse_time(timer) / 1e6)  ## ns / 1e6 -> ms

    # @contextmanager
    # def timeit(self):
    #     t1 = time.time()

    #     yield

    #     t2 = time.time()
    #     self._update((t2 - t1) * 1e3)  ## s * 1e3 -> ms
