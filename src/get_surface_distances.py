import registration
import utils

import argparse
import numpy as np

def pa():
    p = argparse.ArgumentParser(description="Computes surface distances between two meshes")
    p.add_argument('query_vtk', type=str, help="path to vtk file that we are querying")
    p.add_argument('reference_vtk', type=str, help="path to vtk file that we are using as a reference")
    #p.add_argument('outputs_dir', type=str, help="path to output directory from the ICP registration")
    #p.add_argument('--scale', type=bool, default=False, help="whether or not to apply ICP scaling as well")
    #p.add_argument('vtk_seg', type=str, help="path to vtk of segmentation without added markers")
    #p.add_argument('vtk_mark', type=str, help="path to vtk of segmentation without added markers")
    return p.parse_args()


args = pa()
point_mesh = args.query_vtk
surf_mesh = args.reference_vtk

#compute the distances
avg_mse_dists = registration.calc_mse_points_to_mesh(point_mesh, surf_mesh, density_weighted=False)
print(np.sum(avg_mse_dists))