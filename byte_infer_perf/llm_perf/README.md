# Byte LLM Perf

## HOW TO USE

1. define workloads config at `llm_perf/workloads/`
2. implement backend inference code at `llm_perf/backends/`
3. start perf test with following command, e.g. start perf test chatglm with GPU backend 

```bash
python3 byte_infer_perf/llm_perf/core/perf_engine.py --task chatglm-torch-fp16-6b --hardware_type GPU
```