#!/bin/bash -l


dataset=$1


# 
# EM
#
# 5
python3 train.py --routing="em" --dataset="$dataset"  --use_bias=True --layers="32,16,16,16,10" --dimension="8,10,10,10,16" --log_dir="experiments/depth/$dataset/em/bias/5"
python3 train.py --routing="em" --dataset="$dataset" --layers="32,16,16,16,10" --dimension="8,10,10,10,16" --log_dir="experiments/depth/$dataset/em/no_bias/5"

# 6
python3 train.py --routing="em" --dataset="$dataset" --use_bias=True --layers="32,16,16,16,16,10" --dimension="8,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/bias/6"
python3 train.py --routing="em" --dataset="$dataset" --layers="32,16,16,16,16,10" --dimension="8,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/no_bias/6"

# 7
python3 train.py --routing="em" --dataset="$dataset" --use_bias=True --layers="32,16,16,16,16,16,10" --dimension="8,10,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/bias/7"
python3 train.py --routing="em" --dataset="$dataset" --layers="32,16,16,16,16,16,10" --dimension="8,10,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/no_bias/7"

# 8
python3 train.py --routing="em" --dataset="$dataset" --use_bias=True --layers="32,16,16,16,16,16,16,10" --dimension="8,10,10,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/bias/8"
python3 train.py --routing="em" --dataset="$dataset" --layers="32,16,16,16,16,16,16,10" --dimension="8,10,10,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/no_bias/8"

# 9
python3 train.py --routing="em" --dataset="$dataset" --use_bias=True --layers="32,16,16,16,16,16,16,16,10" --dimension="8,10,10,10,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/bias/9"
python3 train.py --routing="em" --dataset="$dataset" --layers="32,16,16,16,16,16,16,16,10" --dimension="8,10,10,10,10,10,10,10,16" --log_dir="experiments/depth/$dataset/em/no_bias/9"


#
# RBA
#
# 2
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=True --layers="32,10" --dimension="8,16" --log_dir="experiments/depth/$dataset/rba/bias/2"
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=False --layers="32,10" --dimension="8,16" --log_dir="experiments/depth/$dataset/rba/no_bias/2"

# 3
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=True --layers="32,16,10" --dimension="8,12,16" --log_dir="experiments/depth/$dataset/rba/bias/3"
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=False --layers="32,16,10" --dimension="8,12,16" --log_dir="experiments/depth/$dataset/rba/no_bias/3"

# 4
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=True --layers="32,16,16,10" --dimension="8,12,12,16" --log_dir="experiments/depth/$dataset/rba/bias/4"
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=False --layers="32,16,16,10" --dimension="8,12,12,16" --log_dir="experiments/depth/$dataset/rba/no_bias/4"

# 5
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=True --layers="32,16,16,16,10" --dimension="8,12,12,12,16" --log_dir="experiments/depth/$dataset/rba/bias/5"
python3 train.py --routing="rba" --dataset="$dataset" --use_bias=False --layers="32,16,16,16,10" --dimension="8,12,12,12,16" --log_dir="experiments/depth/$dataset/rba/no_bias/5"
