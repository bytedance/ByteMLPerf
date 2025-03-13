# ByteMicroPerf

## Introduction
ByteMicroPerf is a part of ByteMLPerf, which is mainly used to evaluate the performance of frequent computation and communication operators in mainstream deep learning models on new emerging heterogeneous hardwares. The main characteristics are as follows:

- Easy and quick access for diverse heterogeneous hardware
- Evaluation process fitting realistic business scenarios
- Coverage of frequent operators across multiple categories

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
python3 launch.py --task exp --hardware_type GPU
```
#### Usage
```
--task: operator name                              please create a workload file for new operators by following the existing style in byte_micro_perf/workloads.

--hardware_type: hardware category name            please derive a Backend class for your heterogeneous hardware in byte_micro_perf/backends.
```

### Expected Output
For different types of operators (Compute-bound / Memory-bound), we adopt various metrics to comprehensively evaluate the performance of the operator. Regarding the various metrics, the explanations are as follows:
| Metric    | Description |
| -------- | ------- |
| Memory Size(MB) | the rough sum of read/write bytes    |
| Kernel bandwidth(GB/s) | the achieved bandwidth under given input size of this kernel     |
| Bandwidth Utilization(%)    | the ratio of achieved bandwidth and theoretical bandwidth   |
| Avg latency(us) |the average of kernel latencies|

Example:
```
{
    "Operator": "EXP",
    "Backend": "GPU",
    "Host Info": "Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz",
    "Device Info": "NVIDIA A800-SXM4-80GB",
    "Performance": [
        {
            "Dtype": "float32",
            "Tensor Shapes": [
                [
                    256,
                    8192
                ]
            ],
            "Read IO Size(MB)": 8.0,
            "Write IO Size(MB)": 8.0,
            "Memory Size(MB)": 16.0,
            "Kernel bandwidth(GB/s)": 1790.52,
            "Bandwidth Utilization(%)": 87.81,
            "Avg latency(us)": 9.37,
            "QPS": 27321.24
        }
    ]
}
```

## Trouble Shooting

For more details, you can visit our offical website here: [bytemlperf.ai](https://bytemlperf.ai/). Please let us know if you need any help or have additional questions and issues!
