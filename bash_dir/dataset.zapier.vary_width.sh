#!/bin/bash

for width in `seq 50 50 1350`;
do
    dir=model/zapier-2/width-$width/
    if [[ ! -e $dir ]]; then
        mkdir -p $dir
    elif [[ ! -d $dir ]]; then
        echo "$dir already exists but is not a directory"
    fi
    CUDA_VISIBLE_DEVICES=1 python train.py --dataset dataset/zapier/zapier_data.pkl --config configs/config_l.json --output $dir  --width $width > $dir/run.zapier.log
done
