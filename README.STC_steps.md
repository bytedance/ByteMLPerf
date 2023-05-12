
1. install hpe 1.5.1
2. create a python virtual environment, install -r ByteMLPerf requirements.txt, activate it
```
python3 -m venv py_venv
source ./py_venv/bin/activate
cd byte_mlperf/backends/STC
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```
3. refer to ByteMLPerf/byte_mlperf prepare_model_and_dataset.sh, download model and dataset
```for example
bash byte_mlperf/prepare_model_and_dataset.sh bert-tf-fp32 open_squad
```
4. export PYTHONPATH=xxx/ByteMLPerf:xxx/ByteMLPerf/byte_mlperf/backends/STC:$PYTHONPATH
5. python3 launch.py --task wdl-tf-fp32 --hardware_type STC, if raise error, don't care it, as long as generate a venv dir in xxx/ByteMLPerf/byte_mlperf/backends/STC
6. source xxx/ByteMLPerf/byte_mlperf/backends/STC/venv/bin/activate
7. stc-ddk commit id: f595bac6, add the change: http://code.streamcomputing.com/solution/stc_ddk/-/merge_requests/171/diffs , then build and install it, make a dir as A to save libstc_runtime.tar.gz
```
cd stc_ddk
git pull origin master
# add the changes manually
bash pkg_build.sh 3.7 ./build debug
cd build
pip3 install stc_ddk-1.0.0-cp37-cp37m-linux_x86_64.whl

mkdir output
mv ../build/libstc_runtime.tar.gz ./
cp -r ../build/libstc_runtime ./
```
8. copy run_engine.cpython-37m-x86_64-linux-gnu.so from 172.16.30.118:/home/qiupeng/projects/ByteMLPerf/run_engine.cpython-37m-x86_64-linux-gnu.so, check its md5sum:ada3cb3def1f8e58affc0729c19e01a8, ldd the so, make sure libstc_runtime.so、libtensorflow_cc.so.1、libonnxruntime.so.1.12.1 link to dir A
```
ldd run_engine.cpython-37m-x86_64-linux-gnu.so
export LD_LIBRARY_PATH=/home/mountpoint/dev-user/chenxin.lv/stc_ddk/output/libstc_runtime/lib:$LD_LIBRARY_PATH
```
9. pip install tb-ubuntu2004==1.11.0 -i https://stcp_user:StcpDownload@sources.streamcomputing.com/simple
10. deactivate
11. python3 launch.py --task wdl-tf-fp32 --hardware_type STC