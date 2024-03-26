# AMD INSTINCT™ MI210 ACCELERATOR

The AMD Instinct™ MI210 accelerator extends AMD industry performance leadership in accelerated compute for double precision (FP64) on PCIe® form factors for mainstream HPC and AI workloads in the data center. Built on AMD Exascale-class technologies with the 2nd Gen AMD CDNA™ architecture, the MI210 enables scientists and researchers to tackle our most pressing challenges from climate change to vaccine research. MI210 accelerators, combined with the AMD ROCm™ 5 software ecosystem, allow innovators to tap the power of HPC and AI data center PCIe® GPUs to accelerate their time to science and discovery.

<font size="4">**Key Features**</font>

|<div style="width:150px">PERFORMANCE</div>|<div style="width:100px">MI210</div>|
| ----------------- |:-----------------------:|
|Compute Units |	104 CU|
|Stream Processors|6,656|
|Matrix Cores|416|
|Peak FP64/FP32 Vector|22.6 TF|
|Peak FP64/FP32 Matrix|45.3 TF|
|Peak FP16/BF16|181.0 TF|
|Peak INT4/INT8|181.0 TOPS|

|<div style="width:150px">MEMORY||
| ----------------- |:-----------------------:|
|Memory Size|	64GB HBM2e|
|Memory Interface|4,096 bits|
|Memory Clock|1.6GHz|
|Memory Bandwidth|up to 1.6 TB/sec|

|<div style="width:150px">RELIABILITY</div>|<div style="width:100px">|
| ----------------- |:----------------------:|
|ECC (Full-chip)|Yes|
|RAS Support|	Yes|

|<div style="width:150px">SCALABILITY</div>|<div style="width:100px">|
| ----------------- |:----------------------:|
|Infinity Fabric<small><small><small>TM</small></small></small> Links|up to 3|
|Coherency Enabled|Yes (Dual &#124; Quad Hives)|
|OS Support|Linux<small><small><small>TM</small></small></small> 64 Bit|
|AMD ROCm<small><small><small>TM</small></small></small>|Yes|
|Compatible|


# Software environment

| Software | Version | Description |
| :-----: | :----: | :----: |
| ROCm | 5.5 | The first open-source software development platform for HPC/Hyperscale-class GPU computing which is built by AMD |
| MIGraphX | build from source | Graph inference engine that accelerates machine learning model inference built by AMD |

# Models supported

| Model name |  Precision | Dataset | Compile time(s) | QPS | P99 Latency |
| ---- | ---- | ---- | ---- | ---- | ---- |
| bert-tf-fp32 | FP32 | Open Squad 1.1 | 69.2 | 295 | 88.56 |
| conformer-encoder-onnx-fp32 | FP32 | Fake Dataset | 752.9 | 185  | 359.89 |
| resnet50-tf-fp32 | FP32 | Open Imagenet | 1,515.8 | 1,945 | 37.93 |
| videobert-onnx-fp32 | FP32 | Open Cifar | 488.4 | 425 | 58.44 |
| widedeep-tf-fp32 | FP32 | Open Criteo Kaggle | 29.34 | 3,169,052  | 5.39 |
| yolov5-onnx-fp32 | FP32 | Fake Dataset | 915.4 | 71  | 970.32 |


# How to run

- Build the docker image

  ```
  docker build .
  ```

- Clone this repo

  ```
  git clone https://github.com/bytedance/ByteMLPerf.git
  ```

- Run byte-mlperf task

  For example,

  ```
  python3 launch.py --task widedeep-tf-fp32 --hardware MIGRAPHX
  ```
