#!/bin/bash

data=../../data/cross_market/processed/*/android/*.p
dataset='android'

# Loop over different setups
for i in {1..20}; do
    python3 ../anomaly.py -l $data -k 10 -m ${i}0 > results/anomaly/cross/$dataset/${i}0.result
done
