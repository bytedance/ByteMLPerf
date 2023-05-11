<div align="center">
  <img src="STC.jpg">
</div>


# Supported model inference results
| Model name | QPS | Dataset | Metric name | Metric value |
| :-----:| :----: | :----: | :----: | :----: |
| bert-tf-fp32 | 820.73 | Open Squad 1.1 | F1 Score | 86.45 |
| bert-torch-fp32 | 807.89 | Open Squad 1.1 | F1 Score | 86.14 |
| resnet50-tf-fp32 | 8728.8 | Open ImageNet | Top-1 | 77.24% |
| widedeep-tf-fp32 | 2418915.33 | Open Criteo Kaggle | Top-1 | 77.39% |


For more detailed result information, see byte_mlperf/reports/STC/. Above model inference based on the chip named "STC P920" and the following software.

| Software | Version | Description |
| :-----:| :----: | :----: |
| HPE | 1.5.1 | Heterogeneous Programming Environment |
| TB | 1.11.0 | TensorTurbo, AI compiler developed based on TVM |
| STC_DDk | 1.1.0 | Model compilation and deployment tools developed based on TensorTurbo |


See the link for more detailed software information: https://docs.streamcomputing.com/zh/latest/

If you are interested in further information about the product, please contact the email: johnson@streamcomputing.com

