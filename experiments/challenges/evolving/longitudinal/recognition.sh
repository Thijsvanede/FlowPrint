#!/bin/bash

# Set data source
# data=../../data/cross_market/processed/*/*/*.p
# dataset="cross/both"
data=../../data/Andrubis/processed/20*.p
dataset="andrubis_filtered"
filter=""

# Loop over different ip fractions
for ip in {0..10}; do
    # Get IP fraction
    frac_ip=`echo "scale=1; $ip /10" |bc -l`

    # Perform analysis on frac_ip
    python3 flow_changes.py -l $data -k 10 -m 100 --fraction-ips $frac_ip --recognition $filter > results/recognition/$dataset/ip_$frac_ip.result
done

# Loop over different certificate fractions
for cert in {0..10}; do
    # Get IP fraction
    frac_cert=`echo "scale=1; $cert /10" |bc -l`

    # Perform analysis on frac_ip
    python3 flow_changes.py -l $data -k 10 -m 100 --fraction-certificates $frac_cert --recognition $filter > results/recognition/$dataset/cert_$frac_cert.result
done

# Loop over different combinations of ip, cert fractions
for ip in {0..10}; do
    # Get IP fraction
    frac=`echo "scale=1; $ip /10" |bc -l`

    # Perform analysis on frac_ip
    python3 flow_changes.py -l $data -k 10 -m 100 --fraction-ips $frac --fraction-certificates $frac --recognition $filter > results/recognition/$dataset/ip_${frac}_cert_${frac}.result
done
