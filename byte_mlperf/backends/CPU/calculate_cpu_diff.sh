#！bin/bash
if [ ! -d "byte_mlperf/backends/CPU/venv" ];then
    virtualenv -p python3 byte_mlperf/backends/CPU/venv
    source byte_mlperf/backends/CPU/venv/bin/activate
    byte_mlperf/backends/CPU/venv/bin/pip3 install --upgrade pip  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    byte_mlperf/backends/CPU/venv/bin/pip3 install -r byte_mlperf/backends/CPU/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
else
    source byte_mlperf/backends/CPU/venv/bin/activate
    byte_mlperf/backends/CPU/venv/bin/pip3 install -r byte_mlperf/backends/CPU/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
fi

python3 byte_mlperf/backends/CPU/calculate_cpu_diff.py --task $1 --batch_size $2
