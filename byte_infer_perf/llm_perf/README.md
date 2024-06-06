# Byte LLM Perf

Vendors can refer to this document for guidance on building backend: [Byte LLM Perf](https://bytedance.larkoffice.com/docx/ZoU7dkPXYoKtJtxlrRMcNGMwnTc)

## Requirements
* Python >= 3.8
* torch==2.1.0

## Installation
```shell
pip3 install torch==2.1.0
pip3 install -r requirements.txt
```

## Quick Start
Please be sure to complete the installation steps before proceeding with the following steps.

To start llm_perf, there are 3 steps:
1. Download opensource model weights(.pt file)
2. Download model output logits in specific input case(.npy file)
3. Start accuracy and performance test case

You can run following command automate all steps with chatglm2 model on GPU backend
```shell
python3 byte_infer_perf/llm_perf/launch.py --hardware_type GPU --task chatglm2-torch-fp16-6b 
```

## Models
The list of supported models is:
* [chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
* [chinese-llama-2-13b](https://huggingface.co/hfl/chinese-llama-2-13b)
* [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
