

dir="/home-local/kimm58/SPIE2023/data/H1Capture/knee"
point_array=("1548" "1909" "1356" "7606" "330")

i=0
find $dir -mindepth 1 -maxdepth 1 -type d -name 'knee*' | while IFS= read -r line
do
    ##For color: make sure to change the python file as well
    # pref=$(echo $line | awk -F '/' '{print $(NF)}')
    # vtk="$line/${pref}.vtk"
    # echo "****** $pref *******"
    # python3 preproc_knee_H1.py $vtk $line

    #For connected Comp 
    pref=$(echo $line | awk -F '/' '{print $(NF)}')
    vtk="$line/${pref}_color.vtk"
    point=${point_array[i]}
    echo "****** $pref *******"
    python3 preproc_knee_H1.py $vtk $line --pid $point
    i=$(($((i))+1))
done 