#!/bin/bash

for width in `seq 5 5 110`;
do
	dir=model/ifttt-2/width-$width
	if [[ ! -e $dir ]]; then
		mkdir -p $dir
	elif [[ ! -d $dir ]]; then
		echo "$dir already exists but is not a directory"
	fi
	CUDA_VISIBLE_DEVICES=0 python train.py --dataset dataset/IFTTT/msr_data.pkl --config configs/config_l.json --output $dir/ --width $width  >$dir/run.ifttt.log
done
