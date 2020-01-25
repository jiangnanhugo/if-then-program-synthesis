#!/bin/bash

for width in `seq 150 50 300`;
do
    dir=model/zapier/width-$width/
    if [[ ! -e $dir ]]; then
        mkdir -p $dir
    elif [[ ! -d $dir ]]; then
        echo "$dir already exists but is not a directory"
    fi
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset dataset/zapier/zapier_data.pkl --config configs/config_l.json --output $dir  --width $width > $dir/run.zapier.log
done
