#!/bin/bash

if [[ -z $1 ]]
then
    echo "Usage"
    exit 0
fi

#1: path to root of data


run_script="/home-local/kimm58/SPIE2023/code/src/run.py"
CT="/home-local/kimm58/SPIE2023/data/KNEE/knee_close.nii.gz"

kneeroot="/home-local/kimm58/SPIE2023/data/NERF/KNEE"

find $kneeroot -maxdepth 1 -mindepth 1 -type d -name 'knee*' | grep -v knee6 | while IFS= read -r line
do
    fpref=$(echo $line | awk -F '/' '{print $(NF)}')
    nerf="$line/${fpref}_color_comp.vtk"
    if [[ ! -e $nerf ]]
    then
        continue
    fi
    #png="$line/inputs/${fpref}.png"
    outdir="$line/outputs"
    if [[ ! -d $outdir ]]
    then
	mkdir $outdir
    fi
    log="$outdir/log.txt"
    if [[ $fpref == "knee6" ]]
    then
        eb="--inv1 True"
    elif [[ $fpref == "knee5" || $fpref == "knee7" ]]
    then
        eb="--inv3 True"
    elif [[ $fpref == "knee2" ]]
    then
        eb="--inv1 True --PCA_input /home-local/kimm58/SPIE2023/data/NERF/KNEE/knee2/outputs/moving_PCA_aligned_testrot.vtk"
    # #else
	# #    eb=""
    # elif [[ $fpref == "hip1" ]]
    # then
	#     eb="--inv1 True --inv2 True"
    fi
    echo "Starting ${fpref} registration..."
    python3 ${run_script} ${CT} ${nerf} x ${outdir} --threshold_value 3000 ${eb} > $log
done

#python3 ../../../code/src/run.py ../../../mask_in/ceramic_hip.nii.gz inputs/hip2.obj inputs/hip2.png outputs/ --threshold_value 200 --inv2 True