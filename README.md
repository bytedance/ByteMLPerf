<div align="center">
  <img src="byte_mlperf/toolutils/icon.png">
</div>


# Byte MLPerf Inference Benchmark Tool
Byte MLPerf (Inference) is a benchmark tool used by ByteDance to assess the performance and accuracy of different models in the inference system in different deployment scenarios. Compared to MLPerf, Byte MLPerf has the following characteristics:
- The model and operating environment will be more representative in the actual business scenario;
- In addition to performance and accuracy, the evaluation of ASIC hardware also includes assessing the ease of use and the extent of graph compilation coverage;
- The performance and accuracy obtained from testing on the Open Model Zoo will serve as a reference for evaluating the introduction of ASIC hardware;

## ByteMLPerf Framework
The architecture of ByteMLPerf is shown in the following diagram:
<div align="center">
  <img src="byte_mlperf/toolutils/flowchat.png">
</div>
The following are described by module

## Installation
Initialize the environment：
```bash
bash init_env.sh
```
Download open model zoo
```bash
cd byte_mlperf & bash prepare_open.sh
```

## Usage
The user uses run.sh as the entry point. When using byte mlperf to evaluate the model, you only need to pass in two parameters --task and --hardware_type, as shown below:
```bash
bash run.sh --tasks xxx --hardware_type xxx
```

1. tasks
--tasks parameter is the name of the incoming workload. The default is all,which will evaluate all workloads, and you can also specify the workload. For example, if you would like to evaluate the workload: open_bert-tf-fp16.json, you need to specify --tasks open_bert-tf-fp16.
Note: All workloads are defined under byte_mlperf/workloads, and the name needs to be aligned with the file name when passing parameters. The current format is open_model-framework-precision.

2. hardware_type
--hardware_type parameter is the incoming hardware_type name, there is no default value, it must be specified by the user. Example: To evaluate Habana Goya, specify --hardware_type GOYA .
Note: All hardware types are defined under byte_mlperf/backends, and the name needs to be aligned with the folder name when passing parameters.

### Workload Description
A workload definition needs to contain the following fields:
```json
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

## Vendor support guide
In the ByteMLPerf system architecture design, the framework and Backend are isolated, and vendors can implement Compile Backend by themselves and participate in the evaluation test as the ByteMLPerf backend.

### Custom Backend
- Create a new folder named after the backend name under the backends/ folder, and all the required dependencies need to be stored in this directory, such as goya backend, and the directory name is GOYA (see the naming rules below for specific naming rules);
- Add backend_xxx.py, where xxx is the backend name, such as goya backend, you need to create an entry file named backend_goya.py, which needs to inherit the class Backend internally;
- Add xxx.json, which is used to interact with users. If there is no need to interact with users, you can provide an empty file and return None directly at get_interact_profile in backend_xxx.py;
- Add requirements.txt, for the required environment dependencies, the framework will create a new venv for each backend, and install the pkg declared in the requirements for it;

Taking Goya as an example, Backend contains the following files:
```bash
byte_mlperf/backends/GOYA/
├── backend_goya.py
├── goya.json
└── requirements.txt
```

### Backend API Description
The Backend base class is declared as follows:
```python
class Backend():
    def __init__(self):
        self.hardware_type = 'UnKnown'
        self.need_reload = False
        self.need_quant = False


    def version(self):
        raise NotImplementedError("Backend:version")

    def pre_optimize(self, configs):
        return None

    def compile(self, configs, dataloader=None):
        raise NotImplementedError("Backend:version")

    # reserve for future
    def tuning(self, configs):
        return None

    # reserve for future
    def segment(self, configs):
        return None

    def get_interact_profile(self, config):
        return None
    
    # Temporary: run the compiled model, return {output_name: value}
    def predict(self, data):
        return None
    
    # Temporary: Performance testing of compiled products
    def benchmark(self, dataloader):
        return None
        
    # Temporary: Returns the Batch Size required for the currently loaded model
    def get_loaded_batch_size(self):
        raise NotImplementedError("Backend: get_loaded_batch_size")
```
- version() : Used to return the version of the currently compiled backend.
- pre_optimize() :Model pre-optimization interface. Requirements: Model pre-optimization cannot change the model format. For example, TF PB should still be TF PB after pre-optimization, and the source framework can still be loaded and run.
- compile() : Model compilation interface. For Vendor that needs to be compiled, model conversion and compilation can be performed here. The model format can be changed here, and the compiled product QS Runtime can be loaded and run. In addition, in addition to returning compiled products, compile also needs to return compilation accuracy, subgraph segmentation information, and compiled model IO information;
```json
result = {
    "model": "ResNet50",
    "framework": "Tensorflow",
    "compile_precision": "int16_mix",
    "input_type": ["INT16"], //a List of String, uppercase needed,
    "max_batch_size" 64, //If multiple batch sizes are not supported, return the currently compiled batch size
    "compile_status": "success", //S uccess should be returned only if all subgraphs are compiled successfully
    "sg_percent": 100, // Compiled op ratio
    "segments": [{
        "sg_idx": 0,
        "is_fallback" : false,
        "input_tensor_map" : {"input:0":[-1,3,255,255]},
        "output_tensor_map" : {"pred:0":[-1,1024]},
        "compiled_model" : [{
            "compiled_bs" : 1,         
            "compiled_obj" : "xxx.obj",
        },],
    },]
}
// As shown in the above example, if multiple subgraphs are compiled, or multiple batch sizes are compiled, they need to be listed separately. It should be noted that the is_fallback field indicates whether the current subgraph will fallback to run on the CPU. If it is true, it usually means that the current subgraph has not been compiled and is split out in the original model format.
```

- tuning() : This interface is reserved for the future. The purpose is that some compilation optimization needs to be improved according to the results of the first compilation and operation. The tuning interface provides such a window for tuning.
- segment() : This interface is reserved for the future, and the purpose is to better adapt to the scene of subgraph compilation in the future. For manufacturers who place segment and compile in the same stage, this interface can be ignored.
- get_interact_profile() : Load the interactive configuration interface. If the vendor needs the user to provide some additional information, you can load the json file you added here and return a list of dict. mlperf will display the content of the profile to the user and is responsible for collecting feedback about the profile. If the user does not need to provide additional information, return None here.
- predict() : Call the compiled product for prediction

## Model Zoo List
Model Zoo&Dataset
The models supported by Byte MlPerf are collected under the Model Zoo. From the perspective of access rights, they are currently divided into internal models and open models. Released with Byte MlPerf is the open model included in the corresponding version.

Open model collection principles:
- Basic Model: including Resnet50, Bert and WnD;
- Business Related Model：Models which are more similar with internal model structures;
- SOTA: including SOTA models corresponding to business domains;

In addition to the complete model structure, Byte MlPerf will also add some typical model substructure subgraphs or OPs (provided that the open model cannot find a suitable model containing such classic substructures), such as transformer encoder/decoder with different sequence lengths , all kinds of common conv ops, such as group conv, depwise-conv, point-wise conv, and rnn common structures, such as gru/lstm, etc.

| Model | Purpose | Framework | Dataset | Precision |
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | regular | tensorflow, pytorch | imagenet2012 | fp32, fp16 |
| bert-base | regular | tensorflow, pytorch | squad-1.1 | fp32, fp16 |
| wide&deep | regular | tensorflow | criteo | fp32 |
| distillbert | business | tensorflow | squad-1.1 | fp32 |
| videobert | business | onnx | cifra100 | fp32 |
| albert | business | pytorch | squad-1.1 | fp32 |
| roformer | business | tensorflow | cail2019 | fp32 |
| vit | business | pytorch | none | fp32 |
| yolov5 | business | onnx | none | fp32 |
| waveglow | business | tensorflow | none | fp32 |
| roberta | sota | pytorch | squad-1.1 | fp32 |
| xlnet | sota | onnx | none | fp32 |
