## Quickstart

### Prepare running environment

```
git clone https://github.com/bytedance/ByteMLPerf.git
cd ByteMLPerf/byte_micro_perf
```

### Prepare hardware configuration(optional)
Please follow the given style at `ByteMLPerf/vendor_zoo` directory to create a new hardware config file for your own heterogeneous hardware. Because this helps the framework evaluate operator performance on new hardware more precisely.

### An example

```
python3 launch.py --task exp --hardware_type MTGPU
```
#### Usage
```
--task: operator name                              please create a workload file for new operators by following the existing style in byte_micro_perf/workloads.

--hardware_type: hardware category name            please derive a Backend class for your heterogeneous hardware in byte_micro_perf/backends.
```