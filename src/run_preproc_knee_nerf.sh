

dir="/home-local/kimm58/SPIE2023/data/NERF/KNEE"
#point_array=("145084" "54726" "219402" "32433")
point_array=("295738" "109478" "192032" "13074")

i=0
find $dir -mindepth 1 -maxdepth 1 -type d -name 'knee*' | grep 'knee2\|knee5\|knee6\|knee7' | while IFS= read -r line
do

    ##For ply to vtk
    # pref=$(echo $line | awk -F '/' '{print $(NF)}')
    # ply="$line/${pref}.ply"
    # if [[ ! -e $ply ]]
    # then
    #     continue
    # fi
    # echo "****** $pref *******"
    # python3 preproc_knee_nerf.py $ply $line


    ##For color: make sure to change the python file as well
    # pref=$(echo $line | awk -F '/' '{print $(NF)}')
    # vtk="$line/${pref}.vtk"
    # echo "****** $pref *******"
    # python3 preproc_knee_nerf.py $vtk $line

    #For connected Comp 
        #6: 145084
        #5: 54726
        #7: 219402
        #2: 32433
    pref=$(echo $line | awk -F '/' '{print $(NF)}')
    vtk="$line/${pref}_color.vtk"
    point=${point_array[i]}
    echo "****** $pref *******"
    python3 preproc_knee_nerf.py $vtk $line --pid $point
    i=$(($((i))+1))
done 