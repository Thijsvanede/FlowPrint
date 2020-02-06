#!/bin/bash

for file in ../data/ReCon/appversions/pcaps/*/*.pcap.*
do
    # Gather capture data
    app=${file#*pcaps/}
    app=${app%/*}
    time=${file#*pcap.}
    id=${file#*$app/}
    id=${id%.pcap*}

    # Split time in date and time
    IFS='-' read -ra time_split <<< "$time"
    date=${time_split[0]}-${time_split[1]}-${time_split[2]}
    time=${time_split[3]}-${time_split[4]}-${time_split[5]}

    # Make output directory
    mkdir -p ../data/ReCon/appversions/processed/$app/$date

    # Create output
    python3 main.py -f $file -s ../data/ReCon/appversions/processed/${app}/${date}/${time}_${id}.p
done
