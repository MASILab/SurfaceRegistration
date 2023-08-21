
outdir="/home-local/kimm58/SPIE2023/data/PaperFigs/surf_dist_vtks"

steak_ref="/home-local/kimm58/SPIE2023/data/H1Capture/steak/steak1/outputs/static_centered.vtk"
hip_ref="/home-local/kimm58/SPIE2023/data/H1Capture/hip1/outputs/static_centered.vtk"
knee_ref="/home-local/kimm58/SPIE2023/data/H1Capture/knee/knee1/outputs/static_centered.vtk"

steakH1="/home-local/kimm58/SPIE2023/data/H1Capture/steak/steak1/outputs/moving_ICP_aligned.vtk"
steakNERF="/home-local/kimm58/SPIE2023/data/NERF/MEAT_TEST/steak2/outputs/moving_ICP_aligned.vtk"

hipH1="/home-local/kimm58/SPIE2023/data/H1Capture/hip1/outputs/moving_ICP_aligned.vtk"
hipNERF="/home-local/kimm58/SPIE2023/data/NERF/HIP_IMPLANT/hip1/outputs/moving_ICP_aligned.vtk"

kneeH1="/home-local/kimm58/SPIE2023/data/H1Capture/knee/knee4/outputs/moving_ICP_aligned.vtk"
kneeNERF="/home-local/kimm58/SPIE2023/data/NERF/KNEE/knee2/outputs/moving_ICP_aligned.vtk"

#python3 create_surf_dist_vtk.py $steakH1 $steak_ref $outdir/steakH1.vtk
#python3 create_surf_dist_vtk.py $steakNERF $steak_ref $outdir/steakNERF.vtk

#python3 create_surf_dist_vtk.py $hipH1 $hip_ref $outdir/hipH1.vtk
#python3 create_surf_dist_vtk.py $hipNERF $hip_ref $outdir/hipNERF.vtk

python3 create_surf_dist_vtk.py $kneeH1 $knee_ref $outdir/kneeH1.vtk
python3 create_surf_dist_vtk.py $kneeNERF $knee_ref $outdir/kneeNERF.vtk