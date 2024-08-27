# ByteMicroPerf 操作说明
# 1、基础信息描述
ByteMicroPerf 是 ByteMLPerf 的一部分，主要用于评估主流深度学习模型在新兴异构硬件上的频繁计算和通信操作的性能。其主要特点如下：
* 方便快捷地支持各种异构硬件
* 评估过程贴合实际业务场景
* 涵盖多个类别的频繁操作符
* 推理精度包括：float32、bfloat16、half、int8


### 功能实现
* 由于天数智芯对cutlass库的维护已经废弃，目前关于矩阵算力Gemm算子相关的int8精度推理主要采用cublas相关的库来实现的，目前通过即时编译的方式实现矩阵算力相关的算子int8精度推理。

### 环境准备：
* sdk版本： 由天数智芯工程师提供
* ixrt版本：由天数智芯工程师提供

# 2、支持的算子列表
### 访存密集型算子
访存密集型算子主要指的是那些在进行计算时需要频繁访问内存的算子，这类算子的执行往往受到内存访问速度的限制，而不是计算能力的限制。访存密集型算子的执行时间很大程度上取决于数据传输的速度>和效率，而不是单纯的计算速度。

* layernorm
* softmax
* reduce
* reduce_min
* reduce_max

### 张量算子
这些算子在张量计算中扮演着重要的角色，它们分别执行不同的操作，以满足各种数据处理和分析的需求。

* index_add
* sort
* unique
* scatter
* gather

### 矩阵运算算子
gemm：它表示一般的矩阵乘法操作，其中矩阵的维度可以是任意的；
gemv：矩阵与向量相乘的操作，它是GEMM的一个特例，其中一个是向量而不是矩阵。GEMV操作在深度学习中的线性层计算中非常常见，因为它能够高效地处理大量的数据。
group_gemm：是一种特殊的矩阵乘法操作，其中多个矩阵被分组并进行并行计算。这种操作在提升计算效率方面非常有效，特别是在处理大规模矩阵运算时，如多专家模型（MoE）的训练中，通过将细碎的专家
计算操作与通信通过Group GEMM算子对多专家计算进行合并，从而提升性能。
batch_gemm：扩展了标准的GEMM操作，允许同时对多个矩阵进行乘法运算。这种操作在需要并行处理多个矩阵乘积的场景中非常有用，例如在深度学习模型的批量处理中。

* gemm
* gemv
* group_gemm
* batch_gemm

### 数据传输算子
在GPU编程中，数据传输涉及到将数据从一个地方（Host，即CPU）移动到另一个地方（Device，即GPU）。

* host2device
* device2host

### 数学运算算子
* sin
* cos
* exp
* exponential
* silu
* gelu
* swiglu
* cast

### 通信算子
这些算子主要用于分布式计算环境中，特别是在大规模并行处理（MPP）和分布式计算领域，它们用于在不同的计算节点之间传输和同步数据。

* allreduce
* allgather
* reducescatter
* alltotal
* broadcast
* p2p

### 二元算子
这些操作涉及两个输入值进行计算。

* add
* mul
* sub
* div

### 注意事项
* 通信类的算子根据workloads下面的配置文件json描述可以实现2、4、8卡直接的通信测试；如果要测试这类算子至少需要两张显卡资源。

# 3、用例测试
### 访存密集型算子
#### layernorm 算子
```bash
python3 launch.py --task layernorm --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### softmax 算子
```bash
python3 launch.py --task softmax --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### reduce_sum 算子
```bash
python3 launch.py --task reduce_sum --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### reduce_min 算子
```bash
python3 launch.py --task reduce_min --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### reduce_max 算子
```bash
python3 launch.py --task reduce_max --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```

### 张量算子
#### index_add 算子
```bash
python3 launch.py --task index_add --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### sort 算子
```bash
python3 launch.py --task sort --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### unique 算子
```bash
python3 launch.py --task unique --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### scatter 算子
```bash
python3 launch.py --task scatter --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### gather 算子
```bash
python3 launch.py --task gather --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```

### 矩阵运算算子
#### gemm 算子
```bash
python3 launch.py --task gemm --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### gemv 算子
```bash
python3 launch.py --task gemv --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### group_gemm 算子
```bash
python3 launch.py --task group_gemm --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### batch_gemm 算子
```bash
python3 launch.py --task batch_gemm --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```

### 数据传输算子
####  host2device 算子
```bash
python3 launch.py --task host2device --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### device2host 算子
```bash
python3 launch.py --task device2host --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```

### 数学运算算子
#### sin 算子
```bash
python3 launch.py --task sin --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### cos 算子
```bash
python3 launch.py --task cos --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### exp 算子
```bash
python3 launch.py --task exp --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### exponential 算子
```bash
python3 launch.py --task exponential --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### silu 算子
```bash
python3 launch.py --task silu --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### gelu 算子
```bash
python3 launch.py --task gelu --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### swiglu 算子
```bash
python3 launch.py --task swiglu --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### cast 算子
```bash
python3 launch.py --task cast --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```

### 通信算子
#### allreduce 算子
```bash
python3 launch.py --task allreduce --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### allgather 算子
```bash
python3 launch.py --task allgather --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### reducescatter 算子
```bash
python3 launch.py --task reducescatter --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### alltotal 算子
```bash
python3 launch.py --task alltotal --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### broadcast 算子
```bash
python3 launch.py --task broadcast --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### p2p 算子
```bash
python3 launch.py --task p2p --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```

### 二元算子
#### add 算子
```bash
python3 launch.py --task add --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### mul 算子
```bash
python3 launch.py --task mul --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### sub 算子
```bash
python3 launch.py --task sub --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
#### div 算子
```bash
python3 launch.py --task div --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json
```
