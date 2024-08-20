import os.path

import onnx
import torch
from onnx2torch import convert


models = ["clip-text-encoder.onnx", "vae-decoder.onnx", "vae-encoder.onnx"]
model_paths = [os.path.abspath(os.path.join(__file__, "../../../stable_diffusion", p)) for p in models]

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

converted_models = [convert(model).to(device) for model in model_paths]
model_inputs = [
    (torch.randint(0, 10, (2, 16), device=device),),
    (torch.randn([2, 4, 32, 32], device=device),),
    (torch.randn([2, 3, 256, 256], device=device),)
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
    out = model(*inputs)
    torch.cuda.synchronize()        
    time_each = time.time() - time_start
    print(f"{name} time is {time_each}")
    if torch.is_tensor(out):
        print(name, out.shape)
    else:
        print(name, [t.shape for t in out])