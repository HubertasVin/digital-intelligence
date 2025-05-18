#!/usr/bin/env bash

DATA_DIR=./rps_data
LOG=results/experiments.log

mkdir -p results
> "$LOG"  # clear previous log

MODEL=simple
BN_FLAG="--batchnorm"
ACT=relu
DROP=0.0
EPOCHS=10

#–– Adam learning‐rate sweep ––
for lr in 1e-4 1e-3 1e-2; do
    echo "===== optimizer=adam lr=${lr} =====" | tee -a "$LOG"
    python rock-paper-scissors.py \
    --data_dir "$DATA_DIR" \
    --model "$MODEL" $BN_FLAG \
    --epochs "$EPOCHS" \
    --optimizer adam \
    --lr "$lr" \
    --activation $ACT \
    --dropout $DROP \
    --output_dir results/lr_adam_${lr} \
    >> "$LOG" 2>&1
    echo "" >> "$LOG"
done

#–– SGD learning‐rate sweep ––
for lr in 1e-4 1e-3 1e-2; do
    echo "===== optimizer=sgd lr=${lr} =====" | tee -a "$LOG"
    python rock-paper-scissors.py \
    --data_dir "$DATA_DIR" \
    --model "$MODEL" $BN_FLAG \
    --epochs "$EPOCHS" \
    --optimizer sgd \
    --lr "$lr" \
    --activation $ACT \
    --dropout $DROP \
    --output_dir results/lr_sgd_${lr} \
    >> "$LOG" 2>&1
    echo "" >> "$LOG"
done

for model in simple vgg resnet; do
    for bn_flag in "" "--batchnorm"; do
        for drop in 0.0 0.25 0.5; do
            echo "===== model=${model} batchnorm=${bn_flag:+on} optimizer=adam lr=1e-4 activation=${act} dropout=${drop} =====" \
            | tee -a "$LOG"
            python rock-paper-scissors.py \
            --data_dir "$DATA_DIR" \
            --model "$model" $bn_flag \
            --epochs 10 \
            --optimizer adam \
            --lr 1e-4 \
            --activation relu \
            --dropout "$drop" \
            --output_dir results/${model}${bn_flag:+_bn}_${optim}_lr${lr}_${act}_drop${drop} \
            --scheduler \
            >> "$LOG" 2>&1
            echo "" >> "$LOG"
        done
    done
done

for activation in relu leaky_relu elu; do
    echo "===== model=resnet batchnorm=on optimizer=adam lr=1e-4 activation=${activation} dropout=${DROP} =====" \
    | tee -a "$LOG"
    python rock-paper-scissors.py \
    --data_dir "$DATA_DIR" \
    --model resnet --batchnorm \
    --epochs "$EPOCHS" \
    --optimizer adam \
    --lr 1e-4 \
    --activation "$activation" \
    --dropout "$DROP" \
    --output_dir results/resnet_adam_lr1e-4_${activation}_drop${DROP} \
    --scheduler \
    >> "$LOG" 2>&1
    echo "" >> "$LOG"
done
