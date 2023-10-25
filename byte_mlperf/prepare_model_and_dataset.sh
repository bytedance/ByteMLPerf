#!/bin/bash

echo "******************* Downloading Model....  *******************"

mkdir -p byte_mlperf/model_zoo/regular
mkdir -p byte_mlperf/model_zoo/popular
mkdir -p byte_mlperf/model_zoo/sota
mkdir -p byte_mlperf/download

#--Basic Model--
# https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/bert_mhlo.tar
# https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/resnet50_mhlo.tar

if [ $1 == "bert-tf-fp32" -o $1 == "bert-torch-fp32" ]; then
    wget -O byte_mlperf/download/open_bert.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_bert.tar
    tar xf byte_mlperf/download/open_bert.tar -C byte_mlperf/model_zoo/regular/
elif [ $1 == "resnet50-tf-fp32" -o $1 == "resnet50-torch-fp32" ]; then
    wget -O byte_mlperf/download/resnet50.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/resnet50.tar
    tar xf byte_mlperf/download/resnet50.tar -C byte_mlperf/model_zoo/regular/
elif [ $1 == "widedeep-tf-fp32" ]; then
    wget -O byte_mlperf/download/open_wide_deep.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_wide_deep_saved_model.tar
    tar xf byte_mlperf/download/open_wide_deep.tar -C byte_mlperf/model_zoo/regular/
#--Popular Model--
elif [ $1 == "albert-torch-fp32" ]; then
    wget -O byte_mlperf/download/open_albert.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_albert.tar
    tar xf byte_mlperf/download/open_albert.tar -C byte_mlperf/model_zoo/popular/ 
elif [ $1 == "roformer-tf-fp32" ]; then
    wget -O byte_mlperf/download/open_roformer.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_roformer.tar
    tar xf byte_mlperf/download/open_roformer.tar -C byte_mlperf/model_zoo/popular/
elif [ $1 == "videobert-onnx-fp32" ]; then
    wget -O byte_mlperf/download/open_videobert.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_videobert.tar
    tar xf byte_mlperf/download/open_videobert.tar -C byte_mlperf/model_zoo/popular/
elif [ $1 == "yolov5-onnx-fp32" ]; then
    wget -O byte_mlperf/download/open_yolov5.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_yolov5.tar
    tar xf byte_mlperf/download/open_yolov5.tar -C byte_mlperf/model_zoo/popular/
elif [ $1 == "conformer-encoder-onnx-fp32" ]; then
    wget -O byte_mlperf/download/open_conformer.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_conformer.tar
    tar xf byte_mlperf/download/open_conformer.tar -C byte_mlperf/model_zoo/popular/
elif [ $1 == "roberta-torch-fp32" ]; then
    wget -O byte_mlperf/download/open_roberta.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_roberta.tar
    tar xf byte_mlperf/download/open_roberta.tar -C byte_mlperf/model_zoo/popular/
elif [ $1 == "deberta-torch-fp32" ]; then
    wget -O byte_mlperf/download/open_deberta.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_deberta.tar
    tar xf byte_mlperf/download/open_deberta.tar -C byte_mlperf/model_zoo/popular/
elif [ $1 == "swin-large-torch-fp32" ]; then
    wget -O byte_mlperf/download/open-swin-large.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open-swin-large.tar
    tar xf byte_mlperf/download/open-swin-large.tar -C byte_mlperf/model_zoo/popular/
#--Sota Model--
elif [ $1 == "vae-encoder-onnx-fp32" -o $1 == "vae-decoder-onnx-fp32" -o $1 == "clip-onnx-fp32" ]; then
    wget -O byte_mlperf/download/stable_diffusion.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/stable_diffusion.tar
    tar xf byte_mlperf/download/stable_diffusion.tar -C byte_mlperf/model_zoo/sota/
elif [ $1 == "unet-onnx-fp32" ]; then
    wget -O byte_mlperf/download/unet.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/unet.tar
    tar xf byte_mlperf/download/unet.tar -C byte_mlperf/model_zoo/sota/
elif [ $1 == "gpt2-torch-fp32" ]; then
    wget -O byte_mlperf/download/traced_gpt2.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/traced_gpt2.tar
    tar xf byte_mlperf/download/gpt2.tar -C byte_mlperf/model_zoo/sota/
elif [ $1 == "chatglm2-6b-torch-fp16" ]; then
    wget -O byte_mlperf/download/chatglm2-6b.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/chatglm2-6b.tar
    tar xf byte_mlperf/download/chatglm2-6b.tar -C byte_mlperf/model_zoo/sota/
elif [ $1 == "llama2-7b-torch-fp16" ]; then
    wget -O byte_mlperf/download/llama-7b.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/llama-7b.tar
    tar xf byte_mlperf/download/llama-7b.tar -C byte_mlperf/model_zoo/sota/
fi

# Download Datasets
if [ $2 == "open_imagenet" ] && [ ! -f "byte_mlperf/download/open_imagenet.tar" ] ; then
    wget -O byte_mlperf/download/open_imagenet.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_imagenet.tar
    tar xf byte_mlperf/download/open_imagenet.tar -C byte_mlperf/datasets/
elif [ $2 == "open_squad" ] && [ ! -f "byte_mlperf/download/open_squad.tar" ]; then
    wget -O byte_mlperf/download/open_squad.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_squad.tar
    tar xf byte_mlperf/download/open_squad.tar -C byte_mlperf/datasets/open_squad
elif [ $2 == "open_criteo_kaggle" ] && [ ! -f "byte_mlperf/download/eval.csv" ]; then
    wget -O byte_mlperf/download/eval.csv https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/eval.csv
    cp byte_mlperf/download/eval.csv byte_mlperf/datasets/open_criteo_kaggle/eval.csv
elif [ $2 == "open_cail2019" ] && [ ! -f "byte_mlperf/download/open_cail2019.tar" ]; then
    wget -O byte_mlperf/download/open_cail2019.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_cail2019.tar
    tar xf byte_mlperf/download/open_cail2019.tar -C byte_mlperf/datasets/open_cail2019 --strip-components 1
elif [ $2 == "open_cifar" ] && [ ! -f "byte_mlperf/download/cifar-100-python.tar" ]; then
    wget -O byte_mlperf/download/cifar-100-python.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/cifar-100-python.tar
    tar xf byte_mlperf/download/cifar-100-python.tar -C byte_mlperf/datasets/open_cifar
fi

echo "Extract Done."
