## 环境安装
```

git clone -b bytemlperf ssh://git@bitbucket.iluvatar.ai:7999/swapp/onnx2torch.git 
cd onnx2torch/
# 模型路径
# ln -s /home/data/bytemlperf/stable_diffusion .
pip3 install onnx onnxconverter onnxconverter_common onnx-simplifier
# 修改为你的路径/path/to/onnx2torch
export PYTHONPATH=${PYTHONPATH}:/path/to/onnx2torch


```
## float32推理
```
python3 tests/models/test_clip_text_encoder.py 
```
## float16推理
```
python3 tests/models/test_clip_text_encoder_half.py 
```