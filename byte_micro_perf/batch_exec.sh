## 访存密集型算子
for task in 'layernorm' 'softmax' 'reduce_sum' 'reduce_min' 'reduce_max'
do
    python3 launch.py --task ${task} --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json;
done

## 张量算子
for task in 'index_add' 'sort' 'unique' 'scatter' 'gather'
do
    python3 launch.py --task ${task} --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json;
done

## 矩阵运算算子
for task in 'gemm' 'gemv' 'group_gemm' 'batch_gemm'
do
    python3 launch.py --task ${task} --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json;
done

## 数据传输算子
for task in 'host2device' 'device2host'
do
    python3 launch.py --task ${task} --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json;
done

## 数学运算算子
for task in 'sin' 'cos' 'exp' 'exponential' 'silu' 'gelu' 'swiglu' 'cast'
do
    python3 launch.py --task ${task} --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json;
done

## 通信算子
for task in 'allreduce' 'allgather' 'reducescatter' 'alltoall' 'broadcast' 'p2p'
do
    python3 launch.py --task ${task} --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json;
done

## 二元算子
for task in 'add' 'mul' 'sub' 'div'
do
    python3 launch.py --task ${task} --hardware_type ILUVATAR --vendor_path ../vendor_zoo/Iluvatar/BI-V150-PCIe.json;
done
