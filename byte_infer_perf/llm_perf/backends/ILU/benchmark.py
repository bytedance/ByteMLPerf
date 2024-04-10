from vllm import LLM, SamplingParams
import torch
import time
import argparse
import os
import csv
from typing import List

'''
{
    "Model": "chinese-llama2-torch-fp16-13b",
    "Backend": "GPU",
    "Host Info": "Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz",
    "Min New Tokens": 128,
    "Max New Tokens": 256,
    "Accuracy": {
        "PPL": [],
        "Token Diff": {},
        "Logits Diff": {}
    },
    "Performance": [
        {
            "TP Size": 1,
            "Batch Size": 1,
            "Input Tokens": 256,
            "First Token Latency(AVG)": 0.2663203875223796,
            "Per Token Latency(AVG)": 0.2794939676920573,
            "First Token Latency(P90)": 0.27227325439453126,
            "Per Token Latency(P90)": 0.2796707717361153,
            "Token Throughput": 3.577788481980987,
            "Request Number": 3,
            "QPS": 0.013921355961015514
        }
    ]
}

'''

class ILUbenchmark():
    def __init__(self, batch, workload, input_tokens, result_queue, max_tokens) -> None:
        self.batch = batch
        self.workload = workload
        self.input_tokens = input_tokens
        self.result_queue =result_queue
        self.max_tokens = max_tokens
        self.config_vllm()
        
        self.ftl = 0
        self.tps = 0
        self.qps = 0

        self.model_path = self.getmodelPath()
        self.init_inference()


    def getmodelPath(self):
        modelpath = "llm_perf/model_zoo/sota/" + self.workload["model"]
        return modelpath


    def getresult(self):        
        return self.input_tokens, self.batch * self.output_token, self.ftl, self.ftl


    def config_vllm(self):
        self.gpu_memory_utilization = 0.9
        self.tensor_parallel_size = 2
        self.output_token = 256
        self.target = None
        self.quantization = None
        self.max_num_seqs = 8

    def init_inference(self):
        self.max_num_seqs = self.max_num_seqs if self.max_num_seqs is not None else self.batch

        self.llm = LLM(model=self.model_path,
          gpu_memory_utilization=self.gpu_memory_utilization,
          max_num_batched_tokens=self.input_tokens * self.max_num_seqs,
          tensor_parallel_size=self.tensor_parallel_size,
          max_num_seqs=self.max_num_seqs,
          trust_remote_code=True,
          quantization=self.quantization,
         ) 		 

        self.sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=self.output_token,)

        self.first_token_sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1,)


    def benchmark_vllm_ftl(self, batch=1, input_tokens=1024):
        print(f'======================= benchmark_vllm_ftl ===============')
        prompt_token_ids = [[0] * input_tokens] * batch
        #outputs = self.llm.generate(sampling_params=self.first_token_sampling_params, prompt_token_ids=prompt_token_ids[:self.max_num_seqs],use_tqdm=False)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        outputs = self.llm.generate(sampling_params=self.first_token_sampling_params, prompt_token_ids=prompt_token_ids)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        duration_time = end_time - start_time

        num_tokens = 0
        for output in outputs:
            assert len(output.outputs[0].token_ids) == 1
            num_tokens += 1
        assert num_tokens == batch
        self.ftl = duration_time


    def benchmark_vllm(self, batch=1, input_tokens=1024):
        print(f'@@@@@@@@@@@@@@@@@@ benchmark_vllm @@@@@@@@@@@@@@@@@@@@')
        prompt_token_ids = [[0] * input_tokens] * batch                

        #outputs = self.llm.generate(sampling_params=self.sampling_params, prompt_token_ids=prompt_token_ids[:self.max_num_seqs],use_tqdm=False)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        outputs = self.llm.generate(sampling_params=self.sampling_params, prompt_token_ids=prompt_token_ids)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        duration_time = end_time - start_time

        num_tokens = 0
        for output in outputs:
            assert len(output.outputs[0].token_ids) == self.output_token
            num_tokens += self.output_token
        assert num_tokens == batch * self.output_token

        model_path = self.model_path.strip()
        model_path = model_path if self.model_path[-1]!='/' else model_path[:-1]
        if self.target is not None:
            val_qps = num_tokens / duration_time
            target_qps = self.target
            if val_qps < target_qps and ((target_qps - val_qps) / target_qps > 0.1 or target_qps - val_qps > 1.5):
                print(f"target qps: {target_qps:.3f}, val qps: {val_qps:.3f}, fail")
                exit(1)
            else:
                print(f"target qps: {target_qps:.3f}, val qps: {val_qps:.3f}, pass")
        else:
            print(f"model: {model_path}, tp: {self.tensor_parallel_size}, batch size: {batch}, input tokens: {self.input_tokens}, output tokens: {self.output_token}, totol output tokens: {batch * self.output_token}, ftl: {self.ftl:.3f} ms, tps: {num_tokens / duration_time :.3f}, requests :{batch / duration_time :.3f} /s, duration_time:{duration_time*1000}ms")
        
        self.qps = num_tokens / duration_time
        self.tps = batch / duration_time
        

