

dir="/home-local/kimm58/SPIE2023/data/NERF/MEAT_TEST"
point_array=("183976" "137171" "137428" "488670" "226058")

i=0
find $dir -mindepth 1 -maxdepth 1 -type d -name 'steak*' | while IFS= read -r line
do

    ##For ply to vtk
    # pref=$(echo $line | awk -F '/' '{print $(NF)}')
    # ply="$line/${pref}.ply"
    # if [[ ! -e $ply ]]
    # then
    #     continue
    # fi
    # echo "****** $pref *******"
    # python3 preproc_steak_nerf.py $ply $line


    ##For color: make sure to change the python file as well
    # pref=$(echo $line | awk -F '/' '{print $(NF)}')
    # vtk="$line/${pref}.vtk"
    # echo "****** $pref *******"
    # python3 preproc_steak_nerf.py $vtk $line

#"183976" "137171" "137428" "488670" "226058"

    #For connected Comp 
        #1: "488670"
        #2: "183976"
        #3: "137171"
        #4: "226058"
        #5: "137428"
    pref=$(echo $line | awk -F '/' '{print $(NF)}')
    vtk="$line/${pref}_color.vtk"
    point=${point_array[i]}
    echo "****** $pref *******"
    python3 preproc_steak_nerf.py $vtk $line --pid $point
    i=$(($((i))+1))
done 