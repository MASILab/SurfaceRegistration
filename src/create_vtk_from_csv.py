import registration
import utils

import argparse
from pathlib import Path

def pa():
    p = argparse.ArgumentParser(description="Registers the surfaces from a CT nifti and a surface file")
    p.add_argument('csv', type=str, help="path to input csv that is output from paraview")
    p.add_argument('vtk', type=str, help="path to output vtk file")
    p.add_argument('outputs_dir', type=str, help="path to output directory from the ICP registration")
    p.add_argument('--scale', type=bool, default=False, help="whether or not to apply ICP scaling as well")
    #p.add_argument('vtk_seg', type=str, help="path to vtk of segmentation without added markers")
    #p.add_argument('vtk_mark', type=str, help="path to vtk of segmentation without added markers")
    return p.parse_args()


args = pa()

csv = Path(args.csv)
out_vtk = Path(args.vtk)
outputs_dir = Path(args.outputs_dir)
reg_vtk = out_vtk.parent/("{}_registered.vtk".format(Path(csv).stem))
scale = args.scale

#create the vtk file
utils.create_vtk_file_from_csv(csv, out_vtk)

#now, register the points

#get the points and the transforms
points = registration.get_fiducials_from_csv(csv)
print(points)
print(type(points))
if scale:
    (pca_r, icp_r, icp_t, icp_scale) = registration.get_transforms(outputs_dir.parent, scale=True)
else:
    (pca_r, icp_r, icp_t) = registration.get_transforms(outputs_dir.parent)

#now apply all the transformations to the points
if scale:
    reg_points = registration.apply_transforms_to_fiducials(pca_r, icp_r, icp_t, points, debug=False, scale=icp_scale, inverse=True)
else:
    reg_points = registration.apply_transforms_to_fiducials(pca_r, icp_r, icp_t, points, debug=False)

#now save these to a new vtk file
registration.create_vtk_pointcloud(reg_points, reg_vtk)