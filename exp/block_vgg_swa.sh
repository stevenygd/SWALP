#! /bin/bash

DATA=${1}
if [ "$DATA" = "CIFAR10" ]; then
    SWALR=0.01
elif [ "$DATA" = "CIFAR100" ]; then
    SWALR=0.01
else
    echo "unknown dataset"
fi
MODEL=VGG16LP
SEED=${2}
python3 train.py \
    --dataset ${DATA} \
    --data_path . \
    --dir block_${MODEL}/${DATA}_${MODEL} \
    --log_name block-${DATA}-${MODEL} \
    --model ${MODEL} \
    --epochs=300 \
    --lr_init=0.05 \
    --swa_start 200 \
    --swa_lr ${SWALR} \
    --wd=5e-4 \
    --seed ${SEED} \
    --save_freq 50 \
    --wl-weight 8 \
    --wl-acc 8 \
    --wl-grad 8 \
    --wl-activate 8 \
    --wl-error 8 \
    --quant-type stochastic \
    --small-block FC \
    --block-dim B;
