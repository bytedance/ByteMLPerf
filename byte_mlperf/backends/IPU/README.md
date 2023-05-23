# INTELLIGENCE PROCESSING UNIT (IPU)

Byte MLPerf supports the Graphcore [Intelligence Processing Unit (IPU)](https://www.graphcore.ai/products/ipu), built for Artificial Intelligence and Machine Learning.

The Graphcore® C600 IPU-Processor card is a dual-slot, full-height PCI Express Gen4 card containing Graphcore’s Mk2 IPU with <b>FP8 support</b>, designed to accelerate machine intelligence applications for <b>both training and inference</b>. All other components are supplied by industry-standard vendors.

The C600 has a thermal design power (TDP) of 185 W running typical workloads and is passively cooled when installed in a suitable chassis enclosure. The maximum power of the card is capped and can be configured to be higher or lower, should that be required.

All the memory on the card is contained within the IPU, providing extremely high bandwidth to the processing cores. There is a total of 900 MB of In-Processor-Memory in the IPU.

The IPU has 1,472 individual machine intelligence cores, generating up to 560 teraFLOPS of FP8 and 280 teraFLOPS of FP16 compute.

The C600 card supports four IPU-Links with a total of 1 Tbps bi-directional bandwidth. C600 cards can be joined together into a cluster of up to eight C600 cards, with each pair of cards linked together with an IPU-Link cable carrying 2 IPU-Links. This gives a much higher IPU-IPU interconnect speed than is available through the PCIe bus alone.

For more information of the Graphcore® C600, please refer to [C600 cards](https://docs.graphcore.ai/en/latest/hardware.html#c600-cards).

# How to access IPUs

To use IPUs you must have access to a system with IPU devices. To get access see [getting started](https://www.graphcore.ai/getstarted).

# PopRT

PopRT is a high-performance inference framework specifically for Graphcore IPUs. It is responsible for deeply optimizing the trained models, generating executable programs that can run on the Graphcore IPUs, and performing low-latency, high-throughput inference.

You can get PopRT and related documents from [graphcore/PopRT](https://github.com/graphcore/PopRT). Docker images are provided at [graphcorecn/poprt](https://hub.docker.com/r/graphcorecn/poprt).

# Models supported

| Model | Domain | Purpose | Framework | Dataset | Precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| albert | nlp | popular | pytorch | squad-1.1 | fp16 |
| bert-base | nlp | regular | pytorch | squad-1.1 | fp16 |
| conformer | nlp | popular | onnx | none | fp16 |
| resnet50-v1.5 | cv | regular | pytorch | imagenet2012 | fp16 |
| roformer | nlp | popular | tensorflow | cail2019 | fp16 |
| wide&deep | rec | regular | tensorflow | criteo | fp16 |

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
