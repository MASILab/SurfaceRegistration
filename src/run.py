import utils
import registration
import argparse

from pathlib import Path

def pa():
    p = argparse.ArgumentParser(description="Registers the surfaces from a CT nifti and a surface file")
    p.add_argument('CT', type=str, help="path to input CT")
    p.add_argument('OBJ', type=str, help="path to input OBJ")
    p.add_argument('PNG', type=str, help="path to input PNG")
    p.add_argument('output_dir', type=str, help="path to output dir")
    p.add_argument('--threshold_value', type=int, default=200, help="Threshold value to use for the CT Mask")
    p.add_argument('--inv1', type=bool, default=False, help="Whether to invert the moving surface along the 1st principle axis")
    p.add_argument('--inv2', type=bool, default=False, help="Whether to invert the moving surface along the 2nd principle axis")
    p.add_argument('--inv3', type=bool, default=False, help="Whether to invert the moving surface along the 3rd principle axis")
    p.add_argument('--isH1', type=bool, default=False, help="Specifies if the surface came from H1 camera or not")
    p.add_argument('--PCA_only', type=bool, default=False, help="Specifies if yo only want to do PCA")
    p.add_argument('--PCA_input', type=str, default=None, help="Path to input mesh that you wish to use for ICP registration instead of the calculated PCA one")
    # p.add_argument('scripts_dir', type=str, help="path to directory where scripts will be")
    # p.add_argument('prequalrootdir', type=str, help="root path to PreQual data folder (on /scratch)")
    # p.add_argument('FreesurferLicense', type=str, help="Path to freesurfer license")
    # #p.add_argument('csv', type=str, help="Path to sub,ses csv")
    # p.add_argument('simg', type=str, help="Path to PreQual simg")
    return p.parse_args()

args = pa()

ct_file = Path(args.CT)
obj_file = Path(args.OBJ)
png_file = Path(args.PNG)
out_dir = Path(args.output_dir)
threshold_value = args.threshold_value
inversions = [args.inv1, args.inv2, args.inv3]
isH1 = args.isH1
PCA_only = args.PCA_only
PCA_input = args.PCA_input

registration.register_vtk_to_nifti(ct_file, obj_file, png_file, out_dir, threshold_value=threshold_value, inversions=inversions, H1=isH1, PCA_only=PCA_only,
input_PCA=PCA_input)