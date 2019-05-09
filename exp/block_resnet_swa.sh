#! /bin/bash

DATA=${1}
MODEL=PreResNet164LP
if [ "$DATA" = "CIFAR10" ]; then
    SWALR=0.01
elif [ "$DATA" = "CIFAR100" ]; then
    SWALR=0.01
else
    echo "unknown dataset"
fi
SEED=${2}
python3 train.py \
    --dataset ${DATA} \
    --data_path . \
    --dir block_${MODEL}/${DATA}_${MODEL} \
    --log_name block-${DATA}-${MODEL} \
    --model ${MODEL} \
    --epochs=225 \
    --lr_init=0.1 \
    --swa_start=150 \
    --swa_lr ${SWALR} \
    --wd=3e-4 \
    --seed ${SEED}\
    --save_freq 50 \
    --wl-weight 8 \
    --wl-grad 8 \
    --wl-activate 8 \
    --wl-error 8 \
    --wl-acc 8 \
    --small-block FC \
    --block-dim B \
    --quant-type stochastic;
