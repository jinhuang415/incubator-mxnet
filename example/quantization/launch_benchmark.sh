# README

# pip install
pip install ../pip/mxnet_mkl_rn50_int8_ww31-1.3.0-py2.py3-none-any.whl

# For CLX-B0

## Launch INT8 Dummy Inference

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=28
# Throughput
numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --benchmark=True
# Latency
numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=1 --num-inference-batches=500 --benchmark=True

## Launch INT8 Real Inference

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=26
# Throughput
numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --batch-size=64 --dataset=../data/ILSVRC2012_img_val.rec --data-nthreads=2
# Latency
numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --batch-size=1 --dataset=../data/ILSVRC2012_img_val.rec --data-nthreads=2

# # For SKX-8180
# 
# ## Launch INT8 Dummy Inference
# 
# export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
# export OMP_NUM_THREADS=28
# # Throughput
# numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=128 --num-inference-batches=500 --benchmark=True
# # Latency
# numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=1 --num-inference-batches=500 --benchmark=True
# 
# ## Launch INT8 Real Inference
# 
# export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
# export OMP_NUM_THREADS=27
# # Throughput
# numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --batch-size=128 --dataset=../data/ILSVRC2012_img_val.rec --data-nthreads=1
# # Latency
# numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --batch-size=1 --dataset=../data/ILSVRC2012_img_val.rec --data-nthreads=1
