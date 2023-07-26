import registration
import utils

import argparse
from pathlib import Path 

def pa():
    p = argparse.ArgumentParser(description="Registers the surfaces from a CT nifti and a surface file")
    p.add_argument('VTK', type=str, help="path to input vtk")
    p.add_argument('output_dir', type=str, help="path to output dir")
    p.add_argument('--pid', type=str, help='Point ID for connected component segmentation')

    return p.parse_args()

args = pa()

vtk_in = Path(args.VTK)
out_dir = Path(args.output_dir)
pref = out_dir.name

# vtk_out = out_dir/("{}_color.vtk".format(pref))
# utils.color_threshold_mesh(vtk_in, vtk_out, abs_threshold=[110,190,190], ratio_threshold=0.75, H1=True)

pid = int(args.pid)
vtk_out = out_dir/("{}_color_comp.vtk".format(pref))
utils.get_component_with_point(vtk_in, vtk_out, point_id=pid)