

dir="/home-local/kimm58/SPIE2023/data/H1Capture/steak"
point_array=("2225" "310" "7749" "6843" "7892")

i=0
find $dir -mindepth 1 -maxdepth 1 -type d -name 'steak*' | while IFS= read -r line
do
    ##For color: make sure to change the python file as well
    # pref=$(echo $line | awk -F '/' '{print $(NF)}')
    # vtk="$line/${pref}.vtk"
    # echo "****** $pref *******"
    # python3 preproc_steak_H1.py $vtk $line

    #For connected Comp
    pref=$(echo $line | awk -F '/' '{print $(NF)}')
    vtk="$line/${pref}_color.vtk"
    point=${point_array[i]}
    echo "****** $pref *******"
    python3 preproc_steak_H1.py $vtk $line --pid $point
    i=$(($((i))+1))
done