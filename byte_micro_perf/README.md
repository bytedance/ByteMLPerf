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
python3 launch.py --task softmax --hardware_type GPU
```
#### Usage
```
--task: operator name                              please create a workload file for new operators by following the existing style in byte_micro_perf/workloads.

--hardware_type: hardware category name            please derive a Backend class for your heterogeneous hardware in byte_micro_perf/backends.

--vendor_path: hardware config path(optional)      it conrresponding to hardware configuration file in ByteMLPerf/vendor_zoo if provided.
```

### Expected Output
```
{
    "Operator": "EXP",
    "Backend": "GPU",
    "Host Info": "Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz",
    "Device Info": "A100-PCIE-40GB",
    "Performance": [
        {
            "Dtype": "float32",
            "Memory Size(MB)": 4.0,
            "Algo bandwidth(GB/s)": 271.83,
            "Bandwidth Utilization(%)": 0.17,
            "Avg latency(us)": 15.43
        }
    ]
}

{
    "Operator": "ALLTOALL",
    "Backend": "GPU",
    "Host Info": "Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz",
    "Device Info": "A100-PCIE-40GB",
    "Performance": [
        {
            "Dtype": "float32",
            "Memory Size(MB)": 0.06,
            "Group": 4,
            "Algo bandwidth(GB/s)": 1.54,
            "Bus bandwidth(GB/s)": 1.15,
            "Bandwidth Utilization(%)": 0.0,
            "Avg latency(us)": 42.58
        }
    ]
}
```

## Trouble Shooting

For more details, you can visit our offical website here: [bytemlperf.ai](https://bytemlperf.ai/). Please let us know if you need any help or have additional questions and issues!
