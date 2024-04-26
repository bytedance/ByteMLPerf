"""
    ****************************************操作说明*********************************
    如果不想跑CPU端的性能、精度、数值指标对比，可以直接执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32（示例）
             如果模型提供了pt、pb格式的优先选择torch的配置进行测试；
             如果执行整个pipeline，需要执行：python3 lauch.py --hardware_type ILUVATAR --task widedeep-tf-fp32（示例）（跑cpu结果会很耗时）

    功能实现：
        1、pt、pb模型转换在compile模块预处理过程中实现；
        2、在天数智芯BI-150显卡上，调用推理引擎tensorrt进行推理，一些onnx模型需要利用前面一步导出的onnx模型再进行插件算子的优化；
    
    环境准备：
        1、sdk版本：由天数智芯工程师提供
        2、ixrt版本：由天数智芯工程师提供

    遗留问题：
        1、roformer模型暂时还不支持动态shape推理，因此本次暂不提交
"""


"""
    ***************************11个小模型的测试与测试报告生成的操作方法****************************
    整个代码运行过程中，主要是从workloads目录下加载对应的模型的配置，主要有test_perf、test_accuracy、test_numeric三项测试内容，用户可以根据自己的需要选择开启与否；
    一般情况下采用字节默认的配置项即可；需要特别修改的配置下面会进行说明

    输出性能文档里面涉及的字段说明：
        1、QPS、AVG Latency、P99 Latency：这3个指标是走字节框架，采用天数智芯的推理引擎IxRT会计算H2D、D2H的时间，也就是数据在不同的设备（CPU、GPU）之间传输耗时；
        2、predict QPS、predict AVG Latency、predict P99 Latency：这部分指标把上面一步计算H2D、D2H的耗时剔除出去了，因此可以看做纯推理耗时，这个耗时可以与利用
           ixerexec命令跑出来的结果做一定的对比，但是不一定完全对齐，因为走整个框架代码肯定会导致一部分性能损失


    cd ByteMLPerf/byte_infer_perf

    1、bert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task bert-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/bert-torch-fp32/

    2、albert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task albert-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/albert-torch-fp32/

    3、debert模型：
           给定的pt模型转成onnx后输入只有2个，因此这里特殊处理了一下；加载处理好的onnx模型：deberta-base-squad-sim_end.onnx
           将其放到：general_perf/model_zoo/popular/open_deberta/ 目录下；

           下载方式：sftp -P 29889 user01@58.247.142.52  密码：5$gS%659
                    cd files/yudefu/ ; get deberta-base-squad-sim_end.onnx

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task deberta-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/deberta-torch-fp32/

    4、roberta模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roberta-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/roberta-torch-fp32/

    5、videobert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task videobert-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/videobert-onnx-fp32
    
    6、widedeep模型：
           该模型经过了特殊的处理，需要采用处理好的onnx模型：widedeep_dynamicshape.onnx；
           将其放到：general_perf/model_zoo/regular/open_wide_deep_saved_model/ 

           下载方式：sftp -P 29889 user01@58.247.142.52  密码：5$gS%659
           cd files/yudefu/ ; get widedeep_dynamicshape.onnx
        
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/widedeep-tf-fp32

    7、swin-transformer模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task swin-large-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/swin-large-torch-fp32

    8、resnet50模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task resnet50-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/resnet50-torch-fp32

    9、yolov5模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task yolov5-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/yolov5-onnx-fp32

    10、conformer模型：
            该onnx模型的transpose算子实现逻辑需要特殊处理；采用处理好的onnx模型：conformer_encoder_optimizer_end.onnx
            将其放到：general_perf/model_zoo/popular/open_conformer/ 

            下载方式：sftp -P 29889 user01@58.247.142.52  密码：5$gS%659
            cd files/yudefu/ ; get conformer_encoder_optimizer_end.onnx
        
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task conformer-encoder-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/conformer-encoder-onnx-fp32

    11、roformer模型：
            该模型暂时没有解决，等待后续解决了再提供测试说明

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roformer-tf-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/roformer-tf-fp32

    12、gpt2模型：
            在进行测试时，请把workloads下面的gpt2-torch-fp32.json里面的精度、数值对比测试改成false

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task gpt2-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/gpt2-torch-fp32
"""

"""
    ***************************大模型操作流程********************
    说明：
        此部分侵入了字节代码框架，因此需要重新重构，暂时不需要进行测试

    操作流程：
        1. 进入ByteMLPerf目录
        2. 执行
            1）python3 byte_infer_perf/llm_perf/core/perf_engine.py --task chatglm2-torch-fp16-6b --hardware_type ILU, 
               得到chatglm2-torch-fp16-6b的精度和性能数据

            2）python3 byte_infer_perf/llm_perf/core/perf_engine.py --task chinese-llama2-torch-fp16-13b --hardware_type ILU,
               得到 chinese-llama2-torch-fp16-13b的精度和性能数据

        3. 在byte_infer_perf/llm_perf/reports/ILU目录下查看得到模型精度和性能数据的json文件
"""
