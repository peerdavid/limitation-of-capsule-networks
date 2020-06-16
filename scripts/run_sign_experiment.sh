#!/bin/bash -l

python3 train_sign.py --routing="em" --use_bias=True --log_dir="experiments/sign/em/bias"
python3 train_sign.py --routing="em" --log_dir="experiments/sign/em/no_bias"

python3 train_sign.py --routing="rba" --use_bias=True --log_dir="experiments/sign/rba/bias"
python3 train_sign.py --routing="rba" --log_dir="experiments/sign/rba/no_bias"
