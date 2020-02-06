#!/bin/bash

data=../../data/Andrubis/processed/20*.p
dataset='andrubis'

# Loop over different setups
for i in {1..100}; do
    python3 ../recognition.py -l $data -k 10 -m ${i}0 -p 100 > results/recognition/$dataset/100/${i}0.result
done
