#!/bin/bash

data=../../data/cross_market/processed/*/ios/*.p
dataset='ios'

# Loop over different setups
for i in {1..20}; do
    python3 ../anomaly.py -l $data -k 10 -m ${i}0 > results/anomaly/cross/$dataset/${i}0.result
done
