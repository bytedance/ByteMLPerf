<div align="center">
  <img src="byte_mlperf/images/icon.png">
</div>


# Byte MLPerf Inference Benchmark Tool
Byte MLPerf(Inference) is an AI Accelerator Benchmark that focuses on evaluating AI Accelerators from practical production perspective, including the ease of use and versatility of software and hardware. Byte MLPerf has the following characteristics:
- Models and runtime environments are more closely aligned with practical business use cases.
- For ASIC hardware evaluation, besides evaluate performance and accuracy, it also measure metrics like compiler usability and coverage.
- Performance and accuracy results obtained from testing on the open Model Zoo serve as reference metrics for evaluating ASIC hardware integration.

Vendors can refer to this document for guidance on building backend: [ByteMLPerf Guide](https://bytedance.us.feishu.cn/docx/L98Mdw3J6obMtJxeRBzuHeRbsof) [[中文版](https://bytedance.feishu.cn/docs/doccno9eLS3OseTA5aMBeeQf2cf#TDK8of)]

## Usage
The user uses launch.py as the entry point. When using byte mlperf to evaluate the model, you only need to pass in two parameters --task and --hardware_type, as shown below:
```bash
python3 launch.py --tasks xxx --hardware_type xxx
```

1. task
--task parameter is the name of the incoming workload. You need to specify the workload. For example, if you would like to evaluate the workload: bert-tf-fp16.json, you need to specify --task bert-tf-fp16.
Note: All workloads are defined under byte_mlperf/workloads, and the name needs to be aligned with the file name when passing parameters. The current format is model-framework-precision.

2. hardware_type
--hardware_type parameter is the incoming hardware_type name, there is no default value, it must be specified by the user. Example: To evaluate Habana Goya, specify --hardware_type GOYA .
Note: All hardware types are defined under byte_mlperf/backends, and the name needs to be aligned with the folder name when passing parameters.

3. compile_only
--compile_only parameter will make task stoped once compilation is finished

4. show_task_list
--show_task_list parameter will print all task name

5. show_hardware_list
--show_hardware_list parameter will print all hardware backend

### Workload Description
A workload definition needs to contain the following fields:
```javascript
{
    "model": "bert-torch-fp32",   //The name of the model to be evaluated, which needs to be aligned with the model_zoo name
    "test_perf": true,            //Evaluate model performance
    "test_accuracy": true,        //Evaluate model accuracy
    "test_numeric": true,         //Accuracy：Evaluate model numeric
    "clients": 3,                 //Performance：Client threads that submit data
    "iterations": 100,            //Performance：How many iterations are submitted by each thread
    "batch_sizes":[1,4,8,16,32,64],//Performance：The batch size when each thread submits data
    "data_percent": 50,           //Accuracy：Ratio of data to assess accuracy, [1-100]
    "compile_only": false,           //Compile the model only
}
```

## Model Zoo List
Model Zoo&Dataset
The models supported by Byte MlPerf are collected under the Model Zoo. From the perspective of access rights, they are currently divided into internal models and open models. Released with Byte MlPerf is the open model included in the corresponding version.

Open model collection principles:
- Basic Model: including Resnet50, Bert and WnD;
- Popular Model：Includes models currently widely used in the industry;
- SOTA: including SOTA models corresponding to business domains;

In addition to the complete model structure, Byte MlPerf will also add some typical model substructure subgraphs or OPs (provided that the open model cannot find a suitable model containing such classic substructures), such as transformer encoder/decoder with different sequence lengths , all kinds of common conv ops, such as group conv, depwise-conv, point-wise conv, and rnn common structures, such as gru/lstm, etc.

| Model | Domain | Purpose | Framework | Dataset | Precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | cv | regular | tensorflow, pytorch | imagenet2012 | fp32 |
| bert-base | nlp | regular | tensorflow, pytorch | squad-1.1 | fp32 |
| wide&deep | rec | regular | tensorflow | criteo | fp32 |
| videobert | mm  |popular | onnx | cifra100 | fp32 |
| albert | nlp | popular | pytorch | squad-1.1 | fp32 |
| conformer | nlp | popular | onnx | none | fp32 |
| roformer | nlp | popular | tensorflow | cail2019 | fp32 |
| yolov5 | cv | popular | onnx | none | fp32 |
| roberta | nlp | popular | pytorch | squad-1.1 | fp32 |
| swin-transformer | cv | popular | pytorch | imagenet2012 | fp32 |
| gpt2 | nlp | sota | pytorch | none | fp32 |
| stable diffusion | cv | sota | onnx | none | fp32 |


## Vendor List
ByteMLPerf Vendor Backend List will be shown below

| Vendor |  SKU | Key Parameters | Supplement |
| ---- | ----| ---- | ---- |
| Intel | Xeon | - | - |
| Stream Computing | STC P920 | Computational power: (fp16)128TFLOPS, Memory Bandwidth: 119.4GB/S | [README.md](byte_mlperf/backends/STC/README.md) |

## Benchmark Summary
Benchmark Result Will be posted here
