#!/bin/bash

if [[ -z $1 ]]
then
    echo "Usage"
    exit 0
fi

#1: path to root of data


run_script="/home-local/kimm58/SPIE2023/code/src/run.py"
CT="/home-local/kimm58/SPIE2023/data/MEAT/steak/steak_w_markers.nii.gz"

steakroot="/home-local/kimm58/SPIE2023/data/NERF/MEAT_TEST"

find $steakroot -maxdepth 1 -mindepth 1 -type d -name 'steak*' | grep steak4 | while IFS= read -r line
do
    fpref=$(echo $line | awk -F '/' '{print $(NF)}')
    nerf="$line/${fpref}_color_comp.vtk"
    #png="$line/inputs/${fpref}.png"
    outdir="$line/outputs"
    if [[ ! -d $outdir ]]
    then
	mkdir $outdir
    fi
    log="$outdir/log.txt"
    if [[ $fpref == "steak1" || $fpref == "steak2" || $fpref == "steak5" ]]
    then
        eb="--inv1 True --inv2 True"
    elif [[ $fpref == "steak3"  ]]
    then
        eb="--inv1 True"
    elif [[ $fpref == "steak4" ]]
    then
        eb="--inv2 True --inv3 True --PCA_input /home-local/kimm58/SPIE2023/data/NERF/MEAT_TEST/steak4/outputs/moving_PCA_aligned_invxz.vtk"
    # #else
	# #    eb=""
    # elif [[ $fpref == "hip1" ]]
    # then
    #     eb="--inv1 True --inv2 True"
    fi
    echo "Starting ${fpref} registration..."
    python3 ${run_script} ${CT} ${nerf} x ${outdir} --threshold_value -500 ${eb} > $log
done

#python3 ../../../code/src/run.py ../../../mask_in/ceramic_hip.nii.gz inputs/hip2.obj inputs/hip2.png outputs/ --threshold_value 200 --inv2 True
