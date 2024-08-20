# ByteMLPerf 操作说明
# 1、基础信息描述

完整的代码框架包括CPU端的性能、精度、数值指标等，是否跑CPU端数据通过workloads里面每一个模型的test_numeric参数控制，并且执行代码需要按照下面的指令发起：python3 lauch.py --hardware_type ILUVATAR --task widedeep-tf-fp32（示例），会比较耗时。

如果不想跑CPU端的性能、精度、数值指标对比，可以直接执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32（示例）

如果模型提供了pt、pb格式的优先选择torch的配置进行测试；

### 功能实现
* pt、pb模型转换在compile模块预处理过程中实现；
* 在天数智芯BI-150显卡上，调用推理引擎tensorrt进行推理，一些onnx模型需要利用前面一步导出的onnx模型再进行插件算子的优化；

### 环境准备：
* sdk版本： 由天数智芯工程师提供
* ixrt版本：由天数智芯工程师提供

# 2、11个常规小模型测试方法
### 数据集、模型准备
```
cd ByteMLPerf/byte_infer_perf/general_perf
bash general_perf/prepare_model_and_dataset.sh bert-torch-fp32 open_squad
bash general_perf/prepare_model_and_dataset.sh resnet50-torch-fp32 open_imagenet
bash general_perf/prepare_model_and_dataset.sh widedeep-tf-fp32 open_criteo_kaggle
bash general_perf/prepare_model_and_dataset.sh albert-torch-fp32
bash general_perf/prepare_model_and_dataset.sh roformer-tf-fp32 open_cail2019
bash general_perf/prepare_model_and_dataset.sh videobert-onnx-fp32 open_cifar
bash general_perf/prepare_model_and_dataset.sh yolov5-onnx-fp32 
bash general_perf/prepare_model_and_dataset.sh conformer-encoder-onnx-fp32
bash general_perf/prepare_model_and_dataset.sh roberta-torch-fp32
bash general_perf/prepare_model_and_dataset.sh deberta-torch-fp32 
bash general_perf/prepare_model_and_dataset.sh swin-large-torch-fp32
bash general_perf/prepare_model_and_dataset.sh gpt2-torch-fp32 

上面的模型下载完毕后会生成在：general_perf/general_perf，需要把该目录在的model_zoo下面的regular、popular、sota移到general_perf/model_zoo目录下。roberta、albert、deberta模型会从huggingface网址下载模型文件，可能遇见访问服务器失败。需要从其他的途径获取。

数据集会生成在：byte_infer_perf/general_perf/datasets/ 目录下，如果依赖的模型数据集下载不完整，会导致推理时报错，各个数据集树形结果如下：
.
├── data_loader.py
├── fake_dataset
│   ├── data_loader.py
│   └── test_accuracy.py
├── open_cail2019
│   ├── data_loader.py
│   ├── pre_process_data.py
│   └── test_accuracy.py
├── open_cifar
│   ├── data_loader.py
│   └── test_accuracy.py
├── open_criteo_kaggle
│   ├── data_loader.py
│   ├── preprocess_dataset.py
│   └── test_accuracy.py
├── open_imagenet
│   ├── data_loader.py
│   └── test_accuracy.py
├── open_squad
│   ├── bert
│   │   ├── accuracy_squad.py
│   │   └── evaluate.py
│   ├── create_squad_data.py
│   ├── data_loader.py
│   └── test_accuracy.py
└── test_accuracy.py

以上的模型、数据集均可以联系天数智芯工程师获取即可。
```
### 性能指标说明
```
整个代码在运行过程中，主要是从workloads目录下加载对应的模型的配置：test_perf、test_accuracy、test_numeric三项测试内容，用户可以根据自己的需要选择开启与否；workloads下面的配置文件修改一般会与modelzoo下面的配置文件保持同步更改。

一般情况下采用字节默认的配置项即可；需要特别修改的配置下面会进行说明。

输出性能文档里面涉及的字段说明：
* QPS、AVG Latency、P99 Latency：这3个指标是字节框架生成的，采用天数智芯的推理引擎IxRT会计算H2D、D2H的时间，也就是数据在不同的设备（CPU、GPU）之间传输耗时；

* predict QPS、predict AVG Latency、predict P99 Latency：这部分指标把上面一步计算H2D、D2H的耗时剔除出去了，因此可以看做纯推理耗时，这个耗时可以与利用ixerexec命令跑出来的结果做一定的对比，但是不一定完全对齐，因为走整个框架代码肯定会导致一部分性能损失。
```

## 支持的模型
### nlp模型
* bert
* albert
* deberta
* videobert
* roberta
* swin-transformer

### 分类与回归模型
* wide&deep

### 分类模型
* renset50

### 检测模型
* yolov5

### 语音识别模型
* conformer

### 预训练语言模型
* roformer

## 测试说明
cd ByteMLPerf/byte_infer_perf

### FP16精度推理
#### bert模型
```bash
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task bert-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/bert-torch-fp32/

注：目前粗略给出最大batch到322
# 更改workloads配置
byte_infer_perf/general_perf/workloads/bert-torch-fp32.json 里面的配置项更改为： "batch_sizes":[1,4,8,16,24,32,48,64,96,128,196,224,322]；
# 更改model_zoo配置
byte_infer_perf/general_perf/model_zoo/bert-torch-fp32.json 配置项更改为："max_batch_size": 322；
# 注意事项
max_batch_size最好与batch_sizes的最大值保持一致，至少不能小于batch_sizes的最大值。
```

#### albert模型
```bash
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task albert-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/albert-torch-fp32/
```

#### debert模型
```bash
给定的pt模型转成onnx后输入只有2个，因此这里特殊处理了一下；加载处理好的onnx模型：deberta-sim-drop-clip-drop-invaild-cast.onnx，移动：mv deberta-sim-drop-clip-drop-invaild-cast.onnx general_perf/model_zoo/popular/open_deberta/
具体获取方式像天数智芯工程师获取

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task deberta-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/deberta-torch-fp32/
```

#### roberta模型
```bash
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roberta-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/roberta-torch-fp32/
```

#### videobert模型
```bash
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task videobert-onnx-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/videobert-onnx-fp32
```

#### widedeep模型
```bash
该模型经过了特殊的处理，需要采用处理好的onnx模型：widedeep_dynamicshape_new.onnx；
将其放到：general_perf/model_zoo/regular/open_wide_deep_saved_model/
移动：mv widedeep_dynamicshape.onnx general_perf/model_zoo/regular/open_wide_deep_saved_model/ 
具体获取方式像天数智芯工程师获取。

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/widedeep-tf-fp32

注：目前粗略测试到最大batch为2000000
# 更改workloads配置
byte_infer_perf/general_perf/workloads/widedeep-tf-fp32.json 配置项更改为：
"batch_sizes":[1024,4096,6000,8000,10000,12000,14000,16384,18000,20000,32200,40000,50000,60000,100000,130000,160000,200000,220000,240000,300000,350000,400000,500000,800000,1000000,1500000,2000000]；
# 更改model_zoo配置
byte_infer_perf/general_perf/model_zoo/widedeep-tf-fp32.json 配置项更改为：
"max_batch_size": 2000000；
# 注意事项
max_batch_size最好与batch_sizes的最大值保持一致，至少不能小于batch_sizes的最大值。
```

#### swin-transformer模型
```bash
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task swin-large-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/swin-large-torch-fp32
```

#### resnet50模型
```bash
# 修改：将general_perf/model_zoo/resnet50-torch-fp32.json 里面的inputs 和 input_shape 中的 "input_1.1" 改为 "input"
## 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task resnet50-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/resnet50-torch-fp32

注：目前粗略测试到最大batch为1300，
# 更改workloads配置
workloads/resnet50-torch-fp32.json配置项更改为："batch_sizes":[1,4,8,16,32,48, 64,82,128,512,1024,1200,1300]；
# 更改model_zoo配置
model_zoo/resnet50-torch-fp32.json 配置项更改为："max_batch_size": 1300；
# 注意事项
max_batch_size 最好与batch_sizes的最大值保持一致，至少不能小于batch_sizes的最大值。
```

#### yolov5模型
```bash
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task yolov5-onnx-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/yolov5-onnx-fp32
```

#### conformer模型
```bash
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task conformer-encoder-onnx-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/conformer-encoder-onnx-fp32
```

#### roformer模型
```bash
该模型经过了特殊的处理，需要采用处理好的onnx模型：roformer_frozen.onnx；
将其放到：general_perf/model_zoo/popular/open_roformer/ 
移动：mv roformer_frozen.onnx general_perf/model_zoo/popular/open_roformer/ 
具体获取方式像天数智芯工程师获取

# 修改：byte_infer_perf/general_perf/model_zoo/roformer-tf-fp32.json里面的inputs及其input_shape，
将两个输入及其输入shape的：冒号去掉
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roformer-tf-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/roformer-tf-fp32
```

### 部分小模型INT8精度推理
```bash
* 目前ixrt推理引擎只实现了部分模型的int8精度推理，因此只提供了下面4个小模型的int8推理case；支持int8推理的模型：resnet50、yolov5、widedeep、bert；
* 注意如果在测试bert的int8推理时，报错，可能是sdk、ixrt版本问题导致，需要升级；
```

#### resnet50模型
```bash
# 更改配置文件
general_perf/model_zoo/resnet50-torch-fp32.json中的model_precision精度为INT8
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task resnet50-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/resnet50-torch-fp32

注：目前粗略测试到最大batch为2000
# 更改workloads配置
byte_infer_perf/general_perf/workloads/resnet50-torch-fp32.json配置项更改为："batch_sizes":[1,4,8,16,32,48,64,82,128,512,1024,1200,1300,1600,2000]；
### 更改model_zoo配置
byte_infer_perf/general_perf/model_zoo/resnet50-torch-fp32.json 配置项更改为："max_batch_size": 2000；
### 注意事项
max_batch_size最好与batch_sizes的最大值保持一致，至少不能小于batch_sizes的最大值。
```


#### widedeep

```bash
# 更改配置文件
general_perf/model_zoo/widedeep-tf-fp32.json 中的 model_precision 精度为 INT8
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/widedeep-tf-fp32

注：目前粗略测试到最大batch为130000
# 更改workloads配置
byte_infer_perf/general_perf/workloads/widedeep-tf-fp32.json配置项更改为："batch_sizes":[1024,4096,6000,8000,10000,12000,13000]；
# 更改model_zoo配置
byte_infer_perf/general_perf/model_zoo/widedeep-tf-fp32.json 配置项更改为："max_batch_size": 130000；
# 注意事项
max_batch_size最好与batch_sizes的最大值保持一致，至少不能小于batch_sizes的最大值。
```


#### yolov5

```bash
# 更改配置文件
general_perf/model_zoo/yolov5-onnx-fp32.json 中的 model_precision 精度为 INT8
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task yolov5-onnx-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/yolov5-onnx-fp32
```


#### bert

```bash
# 更改配置文件
general_perf/model_zoo/bert-torch-fp32.json 中的 model_precision 精度为 INT8，"input_type": "INT32,INT32,INT32"
# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task bert-torch-fp32
# 测试报告位置
general_perf/reports/ILUVATAR/bert-torch-fp32

注：目前粗略给出最大batch到196
# 更改workloads配置
byte_infer_perf/general_perf/workloads/bert-torch-fp32.json，配置项更改为："batch_sizes":[1,4,8,16,24,32,48,64,96,128,196]
# 更改model_zoo配置
byte_infer_perf/general_perf/model_zoo/bert-torch-fp32.json 配置项更改为："max_batch_size": 196
# 注意事项
max_batch_size最好与batch_sizes的最大值保持一致，至少不能小于batch_sizes的最大值。
```

# 3、gpt2模型推理
```bash
# 采用的推理引擎：igie
在进行测试时，请把workloads下面的gpt2-torch-fp32.json里面的精度、数值对比测试改成false；
执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task gpt2-torch-fp32
生成的测试报告位置：general_perf/reports/ILUVATAR/gpt2-torch-fp32
```

# 4、Stable Diffusion模型推理
```bash
# 采用的推理引擎：pytorch
此模块涉及到general_perf下面的vae-decoder、vae-encoder、clip三个模型的推理；

# 环境准备：官方的onnx2torch有bug存在，所以需要安装天数智芯适配版本的onnx2torch，采用pytorch推理框架
cd ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/onnx2torch
执行：pip3 install .

# 数据集、模型准备：
cd ByteMLPerf/byte_infer_perf/general_perf
bash general_perf/prepare_model_and_dataset.sh vae-encoder-onnx-fp32
上面的模型与数据集下载完毕后会生成在：general_perf/general_perf，需要把该目录下的model_zoo下面的regular、popular、sota移到general_perf/model_zoo下面。

# 测试开始
cd ByteMLPerf/byte_infer_perf 
```

#### vae-decoder模型
```bash
注意事项：由于天数智芯的显卡基本上都是32G显存, 因此需要修改workloads下面的模型启动配置为："batch_sizes":[4,8], "test_numeric": false, 

执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task vae-decoder-onnx-fp32
生成的测试报告位置：general_perf/reports/ILUVATAR/vae-decoder-onnx-fp32
```

#### vae-encoder模型
```bash
注意事项：为了实现性能测试, 因此需要修改workloads下面的模型启动配置为："batch_sizes":[4,8], "test_numeric": false, 

执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task clip-onnx-fp32
生成的测试报告位置：general_perf/reports/ILUVATAR/clip-onnx-fp32
```

#### clip模型
```bash
注意事项：为了实现性能测试, 因此需要修改workloads下面的模型启动配置为："test_numeric": false, 

执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task clip-onnx-fp32
生成的测试报告位置：general_perf/reports/ILUVATAR/clip-onnx-fp32
```

# 5、大模型推理
```bash
# 说明：
此部分代码未侵入框架代码，由于vllm框架未实现精度测试，因此精度测试可以沿用GPU的backends；
其次，vllm的TP定义目前与框架定义的tp含义不一样，因此chatglm2、llama2模型的workloads配置里面的TP=2 暂时不考虑，待后续商定好解决方案在继续。

# 环境准备：
需要提前下载天数智芯适配的vllm安装包到测试环境下，为了方便看输出日志，省掉不必要的信息，安装完毕后，
请注释掉：/usr/local/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py 
内部函数async def add_request 下面的logger.info输出日志。

# 测试开始：
cd ByteMLPerf/byte_infer_perf
```

#### chatglm2模型
```bash
执行：python3 llm_perf/launch.py --task chatglm2-torch-fp16-6b --hardware_type ILUVATAR 
生成的测试报告位置：llm_perf/reports/ILUVATAR/chatglm2-torch-fp16-6b
```

#### llama2模型
```bash
执行：python3 llm_perf/launch.py --task chinese-llama2-torch-fp16-13b --hardware_type ILUVATAR
生成的测试报告位置：llm_perf/reports/ILUVATAR/chinese-llama2-torch-fp16-13b
```