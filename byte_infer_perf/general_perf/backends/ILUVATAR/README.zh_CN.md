"""
    操作说明：如果不想跑CPU端的性能、精度、数值指标，可以执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32（示例）;
             如果模型提供了pt、pb格式的优先选择torch的配置进行测试；

    功能实现：
        1、pt、pb模型转换在compile模块预处理过程中实现；
        2、在天数智芯BI-150显卡上，调用推理引擎tensorrt进行推理，一些onnx模型需要利用前面一步导出的onnx模型再进行插件算子的优化；
    
    环境准备：
        1、sdk版本：http://sw.iluvatar.ai/download/corex/daily_packages/latest/x86_64/bi150/sdk/corex-installer-linux64-3.4.0.20240418.74_x86_64_10.2.run
        2、ixrt版本：http://sw.iluvatar.ai/download/corex/daily_packages/latest/x86_64/bi150/apps/py3.10/ixrt-0.9.1+corex.3.4.0.20240418.71-cp310-cp310-linux_x86_64.whl

    遗留问题：
        1、roformer、conformer、widedeep模型做了特殊处理，目前还不能做到加载模型预处理的onnx模型直接进行推理，研发还在继续优化
"""

"""
    ******************下面简单的说明11个小模型是如何测试与测试报告生成的*****************
    整个代码运行过程中，主要是从workloads目录下加载对应的模型的配置，主要有test_perf、test_accuracy、test_numeric三项测试内容，用户可以根据自己的需要选择开启与否；
    一般情况下采用字节默认的配置项即可；

    cd ByteMLPerf/byte_infer_perf;
    1、bert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task bert-torch-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/bert-torch-fp32/

    2、albert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task albert-torch-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/albert-torch-fp32/

    3、debert模型：
           ***给定的pt模型转成onnx后输入只有2个，因此这里特殊处理了一下；目前不能直接使用optimizer脚本优化后的onnx直接进行推理，我们把这个模型优化流程给出了，但是实际上使用了处理好的onnx：
              deberta-base-squad-sim_end.onnx，将其放到：general_perf/model_zoo/popular/open_deberta/ 目录下；
           ***
           其次，需要修改model_zoo下面的general_perf/model_zoo/deberta-torch-fp32.json里面输入的个数，去掉token_type_ids.1相关的配置

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task deberta-torch-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/deberta-torch-fp32/

    4、roberta模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roberta-torch-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/roberta-torch-fp32/

    5、videobert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task videobert-onnx-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/videobert-onnx-fp32
    
    6、widedeep模型：
        ***该模型经过了特殊的处理，需要采用的onnx模型：widedeep_dynamicshape.onnx；将其放到：general_perf/model_zoo/regular/open_wide_deep_saved_model/ 
        ***
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/widedeep-tf-fp32

    7、swin-transformer模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task swin-large-torch-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/swin-large-torch-fp32

    8、resnet50模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task resnet50-torch-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/resnet50-torch-fp32

    9、yolov5模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task yolov5-onnx-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/yolov5-onnx-fp32

    10、conformer模型：
        ***该onnx模型的transpose算子的逻辑是有问题，做了特殊处理；采用处理好的onnx模型：conformer_encoder_optimizer_end.onnx；
           将其放到：general_perf/model_zoo/popular/open_conformer/ 
        ***
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task conformer-encoder-onnx-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/conformer-encoder-onnx-fp32

    11、roformer模型：
        ***********该模型暂时没有解决，等待后续解决了再修改代码，再进行测试***********
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roformer-tf-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/roformer-tf-fp32

    12、gpt2模型：
        *******在进行测试时，请把workloads下面的gpt2-torch-fp32.json里面的精度、数值对比测试改成false
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task gpt2-torch-fp32
        生成的测试报告在：general_perf/reports/ILUVATAR/gpt2-torch-fp32
"""

"""
    ****************大模型操作流程******
    1. 进入ByteMLPerf目录
    2. 执行
        1）python3 byte_infer_perf/llm_perf/core/perf_engine.py --task chatglm2-torch-fp16-6b --hardware_type ILU, 得到chatglm2-torch-fp16-6b的精度和性能数据

        2）python3 byte_infer_perf/llm_perf/core/perf_engine.py --task chinese-llama2-torch-fp16-13b --hardware_type ILU, 得到chinese-llama2-torch-fp16-13b的精度和性能数据

    3. 在byte_infer_perf/llm_perf/reports/ILU目录下查看得到模型精度和性能数据的json文件。
"""
