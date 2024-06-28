# Byte LLM Perf
## Requirements
* Python >= 3.8
* torch >= 2.1.0

## Installation
```shell
# modify according to torch version and hardware
pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# install required packages
pip3 install -r requirements.txt
```

## Quick Start
Please be sure to complete the installation steps before proceeding with the following steps: 
1. Modify task workload, for example, [chatglm2-torch-fp16-6b.json](https://github.com/bytedance/ByteMLPerf/blob/main/byte_infer_perf/llm_perf/workloads/chatglm2-torch-fp16-6b.json)
2. Download model weights using prepare_model.sh or huggingface_cli.
3. Download model output logits in specific input case(.npy files) using prepare_model.sh.
4. Start accuracy and performance tests.

You can run following command automate all steps with chatglm2 model on GPU backend
```shell
python3 byte_infer_perf/llm_perf/launch.py --hardware_type GPU --task chatglm2-torch-fp16-6b 
```

## Demo Project
[GPU Backend](https://github.com/bytedance/ByteMLPerf/tree/main/byte_infer_perf/llm_perf/backends/GPU) provides a demo project that realizes llm inference of chatglm2-6b on A100 with following features: 
- Separate functional components:
    * Scheduler 
        - custom scheduling on tasks
    * Inferencer
        - transfer tasks to real inputs and get outputs
    * Mp Engine
        - deal with TP logic using multiple processes
    * Sampler
        - postprocess logic
    * Ckpt Loader
        - custom ckpt loader with split logic which matches TP logic.
    * Custom model implementation
        - custom model implementation using hardware backend torch realization
- Seperate scheduling logic
    * Context: one task, input_ids shape is [1, q_len]
    * Decode: multiple tasks, input_ids shape up to [max_batch_size, 1]
- Tensor parallelism
- kv cache

The demo project is intended to provide a reference implementation, and there's no guarantee of achieving optimal performance. More technical details will be provided later on [ByteMLPerf](https://bytemlperf.ai)


## Vendor Integration
Vendors can refer to this document for guidance on building backend: [Byte LLM Perf](https://bytemlperf.ai/zh/guide/inference_llm_vendor.html)

## Models
The following models are planned to be supported:
* [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
* [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
* [tiiuae/falcon-180B](https://huggingface.co/tiiuae/falcon-180B)
* [mistralai/Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)

The following models are outdated and will be removed in future vesions:
* [hfl/chinese-llama-2-13b](https://huggingface.co/hfl/chinese-llama-2-13b)

