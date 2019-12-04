#!/bin/bash
width=256
dir=model/ifttt/width-$width
if [[ ! -e $dir ]]; then
    mkdir -p $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory"
fi

#!/bin/bash
for i in `seq 1 10`;
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 nohup python l_train.py --dataset dataset/IFTTT/msr_data.pkl --config configs/config_l.json --output $dir/pid-$i- --width $width  >$dir/run.$i.log &
done
