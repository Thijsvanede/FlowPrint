#!/bin/bash

data=../../data/cross_market/processed/*/android/*.p
dataset='android'

# Loop over different setups
for i in {1..20}; do
    python3 ../recognition.py -l $data -k 10 -m ${i}0 > results/recognition/cross/$dataset/${i}0.result
done
