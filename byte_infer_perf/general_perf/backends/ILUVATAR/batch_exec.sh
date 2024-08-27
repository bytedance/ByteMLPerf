## 批量执行测试代码时，要先执行bash init_environment.sh脚本构建环境；以下的测试只是按照workloads目录下的配置文件来测试案例,
## 如果想更改配置参数请参考：byte_infer_perf/general_perf/backends/ILUVATAR/README.md


cd ByteMLPerf/byte_infer_perf

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task bert-torch-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/bert-torch-fp32/

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task albert-torch-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/albert-torch-fp32/

python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task deberta-torch-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/deberta-torch-fp32/

python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roberta-torch-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/roberta-torch-fp32/

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task videobert-onnx-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/videobert-onnx-fp32

python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/widedeep-tf-fp32

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task swin-large-torch-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/swin-large-torch-fp32

python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task resnet50-torch-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/resnet50-torch-fp32

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task yolov5-onnx-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/yolov5-onnx-fp32

# 执行
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roformer-tf-fp32
# 测试报告位置
# general_perf/reports/ILUVATAR/roformer-tf-fp32
