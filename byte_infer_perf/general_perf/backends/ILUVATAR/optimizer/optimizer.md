# IxRT optimizer

## 1. optimizer 简介
`optimizer` 是一个 ixrt 中集成的图融合工具，用于将onnx图中的op融合成对应的ixrt plugin；

## 2. optimizer 功能说明
| 功能           | 说明  |
| -------------- | ---- |
| 多 batchsize 支持 | 支持设置不同 batchsize 进行推理测试 |
| 动态图支持 | 支持融合动态图和静态图 |
| 模型支持 | 目前测试通过videobert, roberta, deberta, swinL, roformer, albert等模型 |

## 3. optimizer 运行参数
| 参数           | 说明  |
| -------------- | ---- |
| `--onnx`       | 必选 ，指定要运行的 onnx 模型路径 |
| `--num_heads`  | 可选 ，指定模型对应Attention模块注意力头的个数 |
|`--hidden_size`    | 可选， 模型模型隐藏层的大小|
|`--input_shapes` | 可选 ，指定模型输入数据类型，示例 --input_shapes "input_name1:3x224x224, input_name2:3x224x224"类型 |
| `--dump_onnx` | 可选 ，用于图融合过程中dump出中间的onnx图 |
|`--model_type`        | 可选 ，可以指定要融合的模型类型，默认是"bert", 可选["bert", "swint", "roformer"]|
|`--log_level`     |可选 ，指定ixrt运行时显示日志的等级， 可指定为debug、info、error，默认为 info|


## 4. 运行示例

###  4.1 示例1：融合albert|videobert|roberta|deberta
```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH}
```

###  4.2 示例2：融合swinL
```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH} --input_shapes pixel_values.1:${BS}x3x384x384 --model_type swint
```

###  4.3 示例3：融合roformer
```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH} --model_type roformer
```

### 4.4 精度验证

请参考[高级话题](5_advanced_topics.md)中的<u>精度对比工具</u>一节，了解详细使用方法和原理。

也可以用[C++ API 使用简介](3_cpp_api.md)或 [Python API 使用简介](4_python_api.md)

具体使用方法可以参考oss/samples
