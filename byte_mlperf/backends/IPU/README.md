# INTELLIGENCE PROCESSING UNIT (IPU)

Byte MLPerf supports the Graphcore [Intelligence Processing Unit (IPU)](https://www.graphcore.ai/products/ipu), built for Artificial Intelligence and Machine Learning.

# How to access IPUs

To use IPUs you must have access to a system with IPU devices. To get access see [getting started](https://www.graphcore.ai/getstarted).

# PopRT

PopRT is a high-performance inference framework specifically for Graphcore IPUs. It is responsible for deeply optimizing the trained models, generating executable programs that can run on the Graphcore IPUs, and performing low-latency, high-throughput inference.

You can get PopRT and related documents from [graphcore/PopRT](https://github.com/graphcore/PopRT). Docker images are provided at [graphcorecn/poprt](https://hub.docker.com/r/graphcorecn/poprt).

# How to run

- Pull the PopRT docker image

  ```
  docker pull graphcorecn/poprt:latest
  ```

- Enable Poplar SDK

  ```
  source [path-to-sdk]/enable
  popc --version
  ```

- Start docker container

  ```
  gc-docker -- -it \
               -v `pwd -P`:/workspace \
               -w /workspace \
               --entrypoint /bin/bash \
               graphcorecn/poprt:latest
  ```

- Install dependencies in docker container

  ```
  apt-get update && \
  apt-get install wget libglib2.0-0 -y
  ```

- Set environment variable

  ```
  export POPLAR_TARGET_OPTIONS='{"ipuLinkTopology":"line"}'
  ```

- Run byte-mlperf task

  For example,

  ```
  python3 launch.py --task widedeep-tf-fp32 --hardware IPU
  ```
