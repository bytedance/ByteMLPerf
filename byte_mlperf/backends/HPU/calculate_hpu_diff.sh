#！bin/bash
# if [ ! -d "byte_mlperf/backends/HPU/venv" ];then
    # virtualenv -p python3 byte_mlperf/backends/HPU/venv
    # source byte_mlperf/backends/HPU/venv/bin/activate
    # byte_mlperf/backends/HPU/venv/bin/python3 -m pip install --upgrade pip  -q
    # byte_mlperf/backends/HPU/venv/bin/python3 -m pip install -r byte_mlperf/backends/HPU/requirements.txt -q
# else
    # source byte_mlperf/backends/HPU/venv/bin/activate
    # byte_mlperf/backends/HPU/venv/bin/python3 -m pip install -r byte_mlperf/backends/HPU/requirements.txt -q
# fi

python3 byte_mlperf/backends/HPU/calculate_hpu_diff.py --task $1 --batch_size $2
