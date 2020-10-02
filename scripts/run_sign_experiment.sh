#!/bin/bash -l

echo "###################################################"
echo "EM WITHOUT BIAS"
python3 train_sign.py --routing="em"

echo "###################################################"
echo "RBA WITHOUT BIAS"
python3 train_sign.py --routing="rba"

echo "###################################################"
echo "RBA WITH BIAS"
python3 train_sign.py --routing="rba" --use_bias=True

echo "###################################################"
echo "EM WITH BIAS"
python3 train_sign.py --routing="em" --use_bias=True
