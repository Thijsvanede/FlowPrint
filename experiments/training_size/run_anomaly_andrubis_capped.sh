#!/bin/bash

data=../../data/Andrubis/processed/20*.p
dataset='andrubis_capped'

# Loop over different setups
for i in {1..100}; do
    python3 ../anomaly.py -l $data -k 10 -m ${i}0 --cap 500 > results/anomaly/$dataset/${i}0.result
done
