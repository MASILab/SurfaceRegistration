#!/bin/bash

if [[ -z $1 ]]
then
    echo "Usage"
    exit 0
fi

#1: path to root of data


run_script="/home-local/kimm58/SPIE2023/code/src/run.py"
CT="/home-local/kimm58/SPIE2023/data/KNEE/knee_close.nii.gz"

hiproot="/home-local/kimm58/SPIE2023/data/H1Capture/knee"

find $hiproot -maxdepth 1 -mindepth 1 -type d -name 'knee*' | grep knee3 | while IFS= read -r line
do
    fpref=$(echo $line | awk -F '/' '{print $(NF)}')
    #obj="$line/inputs/${fpref}.obj"
    #png="$line/inputs/${fpref}.png"
    vtk_in="$line/${fpref}_color_comp.vtk"
    outdir="$line/outputs"
    if [[ ! -d $outdir ]]
    then
        mkdir $outdir
    fi
    log="$line/outputs/log.txt"
    if [[ $fpref == "knee4" ]]
    then
        eb="--inv1 True"
    elif [[ $fpref == "knee5" ]]
    then
        eb="--inv2 True"
    elif [[ $fpref == "knee2" ]]
    then
        eb="--inv3 True"
    elif [[ $fpref == "knee3" ]]
    then
        eb="--inv3 True"
    # else
	#     eb=""
    fi
    echo "Starting ${fpref} registration..."
    python3 ${run_script} ${CT} ${vtk_in} x ${outdir} --threshold_value 3000 ${eb} --isH1 True > $log
done

#python3 ../../../code/src/run.py ../../../mask_in/ceramic_hip.nii.gz inputs/hip2.obj inputs/hip2.png outputs/ --threshold_value 200 --inv2 True
