#!/bin/bash

if [[ -z $1 ]]
then
    echo "Usage"
    exit 0
fi

#1: path to root of data


run_script="/home-local/kimm58/SPIE2023/code/src/run.py"
CT="/home-local/kimm58/SPIE2023/mask_in/ceramic_hip.nii.gz"

hiproot="/home-local/kimm58/SPIE2023/data/H1Capture"

find $hiproot -maxdepth 1 -mindepth 1 -type d -name 'hip*' | while IFS= read -r line
do
    fpref=$(echo $line | awk -F '/' '{print $(NF)}')
    obj="$line/inputs/${fpref}.obj"
    png="$line/inputs/${fpref}.png"
    outdir="$line/outputs"
    log="$line/outputs/log.txt"
    if [[ $fpref == "hip1" || $fpref == "hip2" ]]
    then
	    eb="--inv2 True"
    elif [[ $fpref == "hip3" ]]
    then
	    eb="--inv3 True"
    elif [[ $fpref == "hip5" ]]
    then
	    eb="--inv2 True --inv3 True"
    else
	    eb=""
    fi
    echo "Starting ${fpref} registration..."
    python3 ${run_script} ${CT} ${obj} ${png} ${outdir} --threshold_value 200 ${eb} > $log
done

#python3 ../../../code/src/run.py ../../../mask_in/ceramic_hip.nii.gz inputs/hip2.obj inputs/hip2.png outputs/ --threshold_value 200 --inv2 True
