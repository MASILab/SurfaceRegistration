import registration
import utils

import argparse
from pathlib import Path
import subprocess

def pa():
    p = argparse.ArgumentParser(description="Registers the surfaces from a CT nifti and a surface file")
    p.add_argument('CT', type=str, help="path to input CT")
    p.add_argument('seg', type=str, help="path to CT original segmentation")
    p.add_argument('outdir', type=str, help="path to output directory")
    #p.add_argument('vtk_seg', type=str, help="path to vtk of segmentation without added markers")
    #p.add_argument('vtk_mark', type=str, help="path to vtk of segmentation without added markers")
    return p.parse_args()

args = pa()

ct = Path(args.CT)
seg = Path(args.seg)
outdir = Path(args.outdir)

marker_seg = outdir/("test_marker.nii.gz")
vtk_seg = seg.parent/(str(seg.name).replace('.nii.gz', '.vtk'))
vtk_mark = marker_seg.parent/(str(marker_seg.name).replace('.nii.gz', '.vtk'))
vtk_seg_moved = vtk_seg.parent/("centered_{}".format(vtk_seg.name))
vtk_mark_moved = vtk_mark.parent/("centered_{}".format(vtk_mark.name))

utils.place_markers_on_seg(ct, seg, marker_seg)
meshcmd = "nii2mesh {} {}".format(str(seg), vtk_seg)
meshcmd_mark = "nii2mesh {} {}".format(str(marker_seg), vtk_mark)
subprocess.run(meshcmd, shell=True)
subprocess.run(meshcmd_mark, shell=True)
utils.align_markered_to_centroid(vtk_seg, vtk_mark, vtk_seg_moved, vtk_mark_moved)




