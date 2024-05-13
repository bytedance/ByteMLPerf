"""
    ***************************大模型操作流程-VLLM框架********************
    说明：
        此部分代码未侵入框架代码，由于vllm框架未实现精度测试，因此精度测试可以沿用GPU的backends；其次，vllm的tp定义目前与框架定义的tp含义不一样，
        因此chatglm2、llama2模型的workloads配置里面的tp=2暂时不考虑，待后续商定好解决方案在继续

    环境准备：
        需要提前下载天数智芯适配的vllm安装包到测试环境下，为了方便看输出日志，省掉不必要的信息，安装完毕后，请注释掉：
        /usr/local/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py 内部函数async def add_request 下面的logger.info输出日志

    数据集模型准备：
        bash prepare_model.sh chatglm2-torch-fp16-6b 注意这里会把chatglm、chatglm2、llama2的数据集、模型都下载下来，我们只需要关注chatglm2、llama2模型，
	在模型这两个模型放到modelzoo/sota 目录下

    测试开始：

    cd ByteMLPerf/byte_infer_perf
        
    1、chatglm2模型：
        执行：python3 llm_perf/launch.py --task chatglm2-torch-fp16-6b --hardware_type ILUVATAR 
        生成的测试报告位置：llm_perf/reports/ILUVATAR/chatglm2-torch-fp16-6b
    
    2、llama2模型：
        执行：python3 llm_perf/launch.py --task chinese-llama2-torch-fp16-13b --hardware_type ILUVATAR
        生成的测试报告位置：llm_perf/reports/ILUVATAR/chinese-llama2-torch-fp16-13b
"""
