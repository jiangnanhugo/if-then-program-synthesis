#!/bin/bash
width=32
dir=model/ifttt/width-$width
if [[ ! -e $dir ]]; then
    mkdir -p $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory"
fi

for i in `seq 1 10`;
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python l_train.py --dataset dataset/IFTTT/msr_data.pkl --config configs/config_l.json --output $dir/pid-$i- --width $width  >$dir/run.$i.log
done
