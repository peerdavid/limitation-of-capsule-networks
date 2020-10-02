#!/bin/bash -l

python3 train_sign.py --routing="em" --use_bias=True
python3 train_sign.py --routing="em"
python3 train_sign.py --routing="rba" --use_bias=True
python3 train_sign.py --routing="rba"
