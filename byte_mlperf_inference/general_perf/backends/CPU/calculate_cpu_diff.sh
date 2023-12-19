#！bin/bash
if [ ! -d "byte_mlperf/backends/CPU/venv" ];then
    virtualenv -p python3 byte_mlperf/backends/CPU/venv
    source byte_mlperf/backends/CPU/venv/bin/activate
    byte_mlperf/backends/CPU/venv/bin/python3 -m pip install --upgrade pip  -q
    byte_mlperf/backends/CPU/venv/bin/python3 -m pip install -r byte_mlperf/backends/CPU/requirements.txt -q
else
    source byte_mlperf/backends/CPU/venv/bin/activate
    byte_mlperf/backends/CPU/venv/bin/python3 -m pip install -r byte_mlperf/backends/CPU/requirements.txt -q
fi

python3 byte_mlperf/backends/CPU/calculate_cpu_diff.py --task $1 --batch_size $2
