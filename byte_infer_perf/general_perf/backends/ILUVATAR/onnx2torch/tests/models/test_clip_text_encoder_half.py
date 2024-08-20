import os.path

import onnx
from onnxsim import simplify
import torch
from onnx2torch import convert
from onnx import load_model, save_model
from onnxmltools.utils import float16_converter
models = ["clip-text-encoder.onnx", "vae-decoder.onnx", "vae-encoder.onnx"]
# models = ["clip-text-encoder.onnx"]

model_paths = [os.path.abspath(os.path.join(__file__, "../../../stable_diffusion", p)) for p in models]

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
#onnx 转成 fp16
onnx_model = [load_model(model) for model in model_paths]
trans_model = [float16_converter.convert_float_to_float16(model,keep_io_types=False) for model in onnx_model]

#onnxsim
for i,model in enumerate(trans_model):
    model_simply, check = simplify(model)
    # onnx.save(model_simply, "new.onnx")
    trans_model[i]=model_simply
    assert check, "Simplified ONNX model could not be validated"
#onnx 转pytorch
converted_models = [convert(model).to(device) for model in trans_model]
dtype=torch.float16
model_inputs = [
    (torch.randint(0, 10, (2, 16), device=device),),    
    (torch.randn([2, 4, 32, 32], device=device,dtype=dtype),),
    (torch.randn([2, 3, 256, 256], device=device,dtype=dtype),)
]
import time
#warmup
for name, model, inputs in zip(models, converted_models, model_inputs):        
    model = model.eval()   
    out = model(*inputs) 
 
for name, model, inputs in zip(models, converted_models, model_inputs):    
    model = model.eval()   
    torch.cuda.synchronize()
    time_start = time.time()
    # torch.cuda.profiler.start()
    out = model(*inputs)
    # torch.cuda.profiler.stop()

    torch.cuda.synchronize()        
    time_each = time.time() - time_start
    print(f"{name} time is {time_each}")
    if torch.is_tensor(out):
        print(name, out.shape)
    else:
        print(name, [t.shape for t in out])