# README

## Launch Quantize

# python imagenet_gen_qsym.py --model=imagenet1k-resnet-50-v1 --calib-dataset=./data/ILSVRC2012_img_val.rec --num-calib-batches=5 --calib-mode=naive --ctx=cpu --quantized-dtype=uint8 --disable-requantize=True --enable-input-calib=True

## Launch INT8 Dummy Inference

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=28
numactl --physcpubind=0-27 --membind=0 python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=128 --num-inference-batches=500 --benchmark=True
