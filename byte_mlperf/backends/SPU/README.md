# Sparse Processing Unit (SPU)

The inventor of the dual sparsity algorithm, Moffett AI has the world's leading sparse computing techniques with more
than 30 patents worldwide. The company creates a new generation of AI computing platform with hardware and software
co-design to achieve order-of-magnitude acceleration of computing performance, reducing latency and low TCO.

The result in Byte MLPerf has demonstrated the potential of sparse computing in inference performance and energy
efficiency, which leads to a lower total cost of ownership (TCO).

For Byte MLPerf, Moffett has submitted performance results of the following models.

| Model                | Precision | QPS     | Dataset        | Metric name | Metric value | 
|----------------------|-----------|---------|----------------|-------------|--------------|
| resnet50-torch-fp32     | INT8      | 59259   | Open Imagenet  | Top-1       | 76.11%      |
| bert-torch-fp32         | INT8/BF16      | 4407 | Open Squad 1.1 | F1 Score    | 86.09     |
| albert-torch-fp32       | INT8/BF16      | 4627 | Open Squad 1.1 | F1 Score    | 87.66      |
| roberta-torch-fp32      | INT8/BF16      | 4389 | Open Squad 1.1 | F1 Score    | 86.63     |
| conformer-encoder-onnx-fp32 | INT8/BF16      | 9329 | Fake Dataset   | Mean Diff   | 0.231      |

Besides the performance results, energy efficiency is another significant highlight of Moffett's devices. For example,
the peak power consumption of S30 is merely
250W.

The Antoum architecture through hardware and software co-design and Moffett's original sparsity algorithm are the
reasons to achieve great performance with high energy efficiency.

The accelerators for AI inference applications in data centers are equipped with Moffett's 1st generation Antoum
processor - the first commercial AI processor with 32x sparsity in the world.

Besides the sparse processing units (SPU) for native sparse convolution and matrix computing in Antoum, the processor
also integrates a Vector Processing Unit (VPU), which enables flexible programmability to keep up with the fast
evolution of AI models.

Also, the on-chip Video Codec, which supports 192-way 1080p video decoding at 30 FPS, and the JPEG decoder, which
supports 1080p image decoding up to 6960 FPS, provide an end-to-end capability for video and image inference workloads.

Moffett provides three SKUs of sparse computing devices, namely S4, S10, and S30. Based on the computing power of S4,
S10 and S30 are designed to be 2 times and 3 times the computing power of S4 respectively.

## How to run

### Prepare Enviroment

Please refer to https://docs.moffettai.com/

### Prepare Model & Datasets

Run `byte_mlperf/prepare_model_and_dataset.sh` to get model and dataset.

### Run Byte MLPerf

For example:

`python3 launch.py --tasks bert-torch-fp32 --hardware_type SPU`

## Contact us

If you are interested in further information about the products, please contact the email: sales@moffett.ai
