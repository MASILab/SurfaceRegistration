import bodyFatSofttissueMask
from pathlib import Path
import os
import argparse
import re

def pa():
        p = argparse.ArgumentParser(description="Registers the surfaces from a CT nifti and a surface file")
        p.add_argument('CT', type=str, help="path to input CT")
        p.add_argument('Outdir', type=str, help="path to output directory")
        p.add_argument('thresh', type=str, help="threshold intensity")
        return p.parse_args()

args = pa()
ct = Path(args.CT)
outdir = Path(args.Outdir)
if not outdir.exists():
    os.mkdir(outdir)
pattern = r'^(.*)\.nii\.gz$'
matches = re.findall(pattern, ct.name)
pref = matches[0]
outfile = outdir/("{}_seg.nii.gz".format(pref))
thresh = int(args.thresh)

bodyFatSofttissueMask.mask_CT(str(ct), str(outfile), threshold_value=thresh)
