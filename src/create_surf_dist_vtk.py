import registration
import utils

import argparse
import pathlib

def pa():
    p = argparse.ArgumentParser(description="Computes surface distances between two meshes")
    p.add_argument('query_vtk', type=str, help="path to vtk file that we are querying")
    p.add_argument('reference_vtk', type=str, help="path to vtk file that we are using as a reference")
    p.add_argument('outputs_vtk', type=str, help="path to output vtk file")

    return p.parse_args()


args = pa()
vtkin = args.query_vtk
vtkref = args.reference_vtk
vtkout = args.outputs_vtk

utils.create_vtk_with_surf_distances(vtkin, vtkref, vtkout)