if [[ -z $1 ]]
then
    echo Usage
    exit 0
fi

#find $1 -name '*centered_fiducials*.csv' | grep pork | grep -v static | while IFS= read -r line
find $1 -name '*fiducials_scaled.csv' | grep -v static | while IFS= read -r line

do
    pref=$(echo $line | awk -F '/' '{print $(NF)}' | cut -d '_' -f 1)
    vtk=$(echo $line | sed -E 's@^(.*).csv$@\1.vtk@g')
    #echo "$line $vtk"
    rootdir=$(dirname $line)
    outsdir="$rootdir/$pref/outputs"

    python3 create_vtk_from_csv.py $line $vtk $outsdir --scale True
done