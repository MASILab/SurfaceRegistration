
if [[ -z $1 ]]
then
    echo Usage
    exit 1
fi


#find $1 -name '*ICP_aligned.vtk' | grep -v knee | grep -v pork | grep -v hip | while IFS= read -r line
find $1 -name '*ICP_aligned.vtk' | grep pork | grep -v pork2 | while IFS= read -r line
do
    fpref=$(echo $line | awk -F '/' '{print $(NF-2)}')
    outdir=$(dirname $line)
    static="$outdir/static_centered.vtk"

    echo "$fpref avg MSE surface distance:"
    python3 get_surface_distances.py $line $static

done