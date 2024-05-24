"""
    ****************************************操作说明*********************************
    如果不想跑CPU端的性能、精度、数值指标对比，可以直接执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32（示例）
             如果模型提供了pt、pb格式的优先选择torch的配置进行测试；
             如果执行整个pipeline，需要执行：python3 lauch.py --hardware_type ILUVATAR --task widedeep-tf-fp32（示例）（跑cpu结果会很耗时）

    功能实现：
        1、pt、pb模型转换在compile模块预处理过程中实现；
        2、在天数智芯BI-150显卡上，调用推理引擎tensorrt进行推理，一些onnx模型需要利用前面一步导出的onnx模型再进行插件算子的优化；
    
    环境准备：
        1、sdk版本： 由天数智芯工程师提供
        2、ixrt版本：由天数智芯工程师提供
"""


"""
    ***************************11个小模型的测试与测试报告生成的操作方法****************************
    整个代码运行过程中，主要是从workloads目录下加载对应的模型的配置，主要有test_perf、test_accuracy、test_numeric三项测试内容，用户可以根据自己的需要选择开启与否；
    一般情况下采用字节默认的配置项即可；需要特别修改的配置下面会进行说明

    输出性能文档里面涉及的字段说明：
        1、QPS、AVG Latency、P99 Latency：这3个指标是走字节框架，采用天数智芯的推理引擎IxRT会计算H2D、D2H的时间，也就是数据在不同的设备（CPU、GPU）之间传输耗时；
        2、predict QPS、predict AVG Latency、predict P99 Latency：这部分指标把上面一步计算H2D、D2H的耗时剔除出去了，因此可以看做纯推理耗时，这个耗时可以与利用
           ixerexec命令跑出来的结果做一定的对比，但是不一定完全对齐，因为走整个框架代码肯定会导致一部分性能损失

    数据集、模型准备：
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

        上面的模型与数据集下载完毕后会生成在：general_perf/general_perf，需要把该目录在的model_zoo下面的regular、popular、sota移到general_perf/model_zoo下面
        如果还缺少什么模型、数据集可以在prepare_model_and_dataset.sh里面执行类似上面的操作即可；


    测试开始：

    cd ByteMLPerf/byte_infer_perf

    备注：由于sftp机器崩溃，文件全部丢失，因此已有的获取数据方式可能不存在了

    1、bert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task bert-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/bert-torch-fp32/

    2、albert模型：
        测试过程中如果从huggingface网址不能下载文件，可以按照下面的操作进行下载
        
        下载方式：sftp -P 29880 vipzjtd@iftp.iluvatar.com.cn（如果链接不上用ip替换：10.160.20.60）  密码：123..com
                 get /upload/3-app/byteperf/madlag.tar
                 tar -zxvf madlag.tar
                 exit

        接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py
        AutoTokenizer.from_pretrained("madlag/albert-base-v2-squad") => AutoTokenizer.from_pretrained("/ByteMLPerf/byte_infer_perf/madlag/albert-base-v2-squad")  (注意绝对路径根据实际情况修改，需要在ByteMLPerf前面在加一个当前目录最上层的路径，下同)

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task albert-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/albert-torch-fp32/

    3、debert模型：
        测试过程中如果从huggingface网址不能下载文件，可以按照下面的操作进行下载

        下载方式：sftp -P 29880 vipzjtd@iftp.iluvatar.com.cn（如果链接不上用ip替换：10.160.20.60）  密码：123..com
                 get /upload/3-app/byteperf/Palak.tar
                 tar -zxvf Palak.tar
                 exit

        接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py
        AutoTokenizer.from_pretrained("Palak/microsoft_deberta-base_squad") => AutoTokenizer.from_pretrained("/ByteMLPerf/byte_infer_perf/Palak/microsoft_deberta-base_squad")

        给定的pt模型转成onnx后输入只有2个，因此这里特殊处理了一下；加载处理好的onnx模型：deberta-sim-drop-clip-drop-invaild-cast.onnx
        将其放到：general_perf/model_zoo/popular/open_deberta/ 目录下；

        下载方式：sftp -P 29889 user01@58.247.142.52  密码：5$gS%659
                 cd yudefu/bytedance_perf ; get deberta-sim-drop-clip-drop-invaild-cast.onnx
                 exit
        
        移动：mv deberta-sim-drop-clip-drop-invaild-cast.onnx general_perf/model_zoo/popular/open_deberta/

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task deberta-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/deberta-torch-fp32/

    4、roberta模型：
        测试过程中如果从huggingface网址不能下载文件，可以按照下面的操作进行下载

        下载方式：sftp -P 29880 vipzjtd@iftp.iluvatar.com.cn（如果链接不上用ip替换：10.160.20.60）  密码：123..com
                 get /upload/3-app/byteperf/csarron.tar
                 tar -zxvf csarron.tar
                 exit

        接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py
        AutoTokenizer.from_pretrained("csarron/roberta-base-squad-v1") => AutoTokenizer.from_pretrained("/ByteMLPerf/byte_infer_perf/csarron/roberta-base-squad-v1")

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roberta-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/roberta-torch-fp32/

    5、videobert模型：
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task videobert-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/videobert-onnx-fp32
    
    6、widedeep模型：
        该模型经过了特殊的处理，需要采用处理好的onnx模型：widedeep_dynamicshape_new.onnx；
        将其放到：general_perf/model_zoo/regular/open_wide_deep_saved_model/ 

        下载方式：sftp -P 29889 user01@58.247.142.52  密码：5$gS%659
                 cd yudefu/bytedance_perf ; get widedeep_dynamicshape_new.onnx
                 exit
        
        移动：mv widedeep_dynamicshape_new.onnx general_perf/model_zoo/regular/open_wide_deep_saved_model/ 
        
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
        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task conformer-encoder-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/conformer-encoder-onnx-fp32

    11、roformer模型：
        该模型经过了特殊的处理，需要采用处理好的onnx模型：roformer_frozen.onnx；
        将其放到：general_perf/model_zoo/popular/open_roformer/ 

        下载方式：sftp -P 29889 user01@58.247.142.52  密码：5$gS%659
                 cd yudefu/bytedance_perf ; get roformer_frozen.onnx
                 exit
        
        移动：mv roformer_frozen.onnx general_perf/model_zoo/popular/open_roformer/ 
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

"""
    ***************************Stable Diffusion模型操作流程********************
    环境准备：官方的onnx2torch有bug存在，所以需要安装天数智芯适配版本的onnx2torch，采用pytorch推理框架

    操作过程：
        1、cd ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/onnx2torch
        2、执行：python3 setup.py install
        3、cd -

        数据集、模型准备：
        cd ByteMLPerf/byte_infer_perf/general_perf

        bash general_perf/prepare_model_and_dataset.sh vae-encoder-onnx-fp32

        上面的模型与数据集下载完毕后会生成在：general_perf/general_perf，需要把该目录在的model_zoo下面的regular、popular、sota移到general_perf/model_zoo下面
        如果还缺少什么模型、数据集可以在prepare_model_and_dataset.sh里面执行类似上面的操作即可；

    测试开始：

    cd ByteMLPerf/byte_infer_perf

    1、vae-decoder模型:
        注意事项：由于天数智芯的显卡基本上都是32G显存, 因此需要修改workloads下面的模型启动配置
            "batch_sizes":[4,8], "test_numeric": false, 

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task vae-decoder-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/vae-decoder-onnx-fp32

    2、vae-encoder模型：
        注意事项：由于天数智芯的显卡基本上都是32G显存, 因此需要修改workloads下面的模型启动配置
            "batch_sizes":[4,8], "test_numeric": false, 

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task vae-encoder-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/vae-encoder-onnx-fp32

    2、clip模型：
        注意事项：为了实现性能测试, 因此需要修改workloads下面的模型启动配置
            "test_numeric": false, 

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task clip-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/clip-onnx-fp32
"""


"""
    ***************************大模型操作流程-VLLM框架********************
    说明：
        此部分代码未侵入框架代码，由于vllm框架未实现精度测试，因此精度测试可以沿用GPU的backends；其次，vllm的tp定义目前与框架定义的tp含义不一样，
        因此chatglm2、llama2模型的workloads配置里面的tp=2暂时不考虑，待后续商定好解决方案在继续

    环境准备：
        需要提前下载天数智芯适配的vllm安装包到测试环境下，为了方便看输出日志，省掉不必要的信息，安装完毕后，请注释掉：
        /usr/local/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py 内部函数async def add_request 下面的logger.info输出日志

    测试开始：

    cd ByteMLPerf/byte_infer_perf
        
    1、chatglm2模型：
        执行：python3 llm_perf/launch.py --task chatglm2-torch-fp16-6b --hardware_type ILUVATAR 
        生成的测试报告位置：llm_perf/reports/ILUVATAR/chatglm2-torch-fp16-6b
    
    2、llama2模型：
        执行：python3 llm_perf/launch.py --task chinese-llama2-torch-fp16-13b --hardware_type ILUVATAR
        生成的测试报告位置：llm_perf/reports/ILUVATAR/chinese-llama2-torch-fp16-13b
"""


"""
    **************************部分小模型的int8精度推理测试************************
    说明：
        字节目前想验证部分小模型的int8精度推理的性能，因此需要基于ixrt（tensorrt）推理引擎进行适配支持
        目前需要验证的小模型包括：resnet50、yolov5、widedeep、bert

        注意如果在测试bert的int8推理时，报错，可能是sdk、ixrt版本问题导致；需要升级；
        生成的报告，并没有更改里面的精度标识，这里只是给出一个测试case，因此并没有将这部分代码加到代码中
    
    环境准备：不需要特别准备，之前如果测试过小模型的性能，相关的环境已经存在了；

    测试开始：

    cd ByteMLPerf/byte_infer_perf

    1、resnet50 模型：
        模型准备：在进行int8精度推理时，需要提供经过量化后的onnx模型，这里直接给出量化好的模型

        下载方式：
            sftp -P 29889 user01@58.247.142.52  密码：5$gS%659（内网连接：sftp -P 29889 user01@10.160.20.61）
            cd yudefu/bytedance_perf  
            get quantized_Resnet50.onnx  
            exit
            mv quantized_Resnet50.onnx general_perf/model_zoo/regular/open_resnet50

        手动更改配置文件：general_perf/model_zoo/resnet50-torch-fp32.json 中的 model_precision 精度为 INT8

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task resnet50-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/resnet50-torch-fp32

    2、widedeep 模型：
        模型准备：在进行int8精度推理时，需要提供经过量化后的onnx模型，这里直接给出量化好的模型

        下载方式：
            sftp -P 29889 user01@58.247.142.52  密码：5$gS%659（内网连接：sftp -P 29889 user01@10.160.20.61）
            cd yudefu/bytedance_perf  
            get quantized_widedeep_staticshape.onnx 
            exit
            mv quantized_widedeep_staticshape.onnx general_perf/model_zoo/regular/open_wide_deep_saved_model/

        手动更改配置文件：general_perf/model_zoo/widedeep-tf-fp32.json 中的 model_precision 精度为 INT8

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/widedeep-tf-fp32

    3、yolov5 模型：
        模型准备：在进行int8精度推理时，需要提供经过量化后的onnx模型，这里直接给出量化好的模型

        下载方式：
            sftp -P 29889 user01@58.247.142.52  密码：5$gS%659（内网连接：sftp -P 29889 user01@10.160.20.61）
            cd yudefu/bytedance_perf  
            get quantized_yolov5s.onnx 
            exit
            mv quantized_yolov5s.onnx general_perf/model_zoo/popular/open_yolov5/

        手动更改配置文件：general_perf/model_zoo/yolov5-onnx-fp32.json 中的 model_precision 精度为 INT8

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task yolov5-onnx-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/yolov5-onnx-fp32

    4、bert 模型：
        模型准备：在进行int8精度推理时，需要提供经过量化后的onnx模型，这里直接给出量化好的模型；该模型直接拿生成好的engine进行推理

        下载方式：
            sftp -P 29889 user01@58.247.142.52  密码：5$gS%659（内网连接：sftp -P 29889 user01@10.160.20.61）
            cd yudefu/bytedance_perf  
            get bert_zijie_int8_b196.engine  
            exit
            mv bert_zijie_int8_b196.engine general_perf/model_zoo/regular/open_bert/

        手动更改配置文件：general_perf/model_zoo/bert-torch-fp32.json 中的 model_precision 精度为 INT8

        执行：python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task bert-torch-fp32
        生成的测试报告位置：general_perf/reports/ILUVATAR/bert-torch-fp32
"""