from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import torch
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds
import vtk
import trimesh
import numpy as np
import subprocess

import utils
import bodyFatSofttissueMask

###
### Compute the MSE distance between two surfaces
###
## given a surface and a list of points, find the closest point on the mesh

def load_vtk_as_trimesh(vtk_file):
    #ASSUMES that the vtk file has faces and is not just a point cloud
    #reads in a vtk file to get the vertices and faces
        #returns a trimesh object consisting of the vertices and faces

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()

    # Get the vertex positions from the VTK dataset
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    #also get the faces
    faces = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        face = []
        for j in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(j)
            face.append(point_id)
        faces.append(face)

    return trimesh.Trimesh(vertices=vertices, faces=faces)

def compute_closest_points_on_surface(point_mesh, surf_mesh):
    #given two vtk files, computes the closest points on the surface of one mesh to each point on the other mesh
        #point_mesh: file that cointains the points we wish to query
        #surf_mesh: file that contains the surface in question

    #read in the points we wish to query
    points = load_vtk_as_trimesh(point_mesh).vertices
    #print(points.shape)

    #read in the surface we wish to query
    mesh = load_vtk_as_trimesh(surf_mesh)
    #print(mesh.vertices.shape)
    
    #now, compute the closest points on "mesh" to each point in "points"
    (closest_points, distances, triangle_id) = mesh.nearest.on_surface(points)

    return (closest_points, distances, triangle_id)

def calc_mse_points_to_mesh(point_mesh, surf_mesh):
    #given two VTK files, compute the MSE distances from the points of one mesh to the surface of another
        #point_mesh: file that cointains the points we wish to query
        #surf_mesh: file that contains the surface in question
    
    #get the distances
    (closest_points, distances, triangle_id) = compute_closest_points_on_surface(point_mesh, surf_mesh)

    #now square them
    return np.array(distances)**2


####
# def plot_mesh_and_pricipal_axes(vertices, centroid, eigenvalues, principal_axes):
#     # Plot the mesh
#     fig = go.Figure(data=[go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], color='lightblue')])

#     # Plot the principal axes
#     origin = centroid
#     axes_lengths = eigenvalues  # You can scale the axes by the eigenvalues if desired

#     # Compute the maximum distance between any two points in the mesh
#     max_distance = np.max(np.linalg.norm(vertices - centroid, axis=1))

#     # Scale factor for the axes
#     scale_factor = max_distance / np.max(eigenvalues)

#     # Plot the first principal axis
#     axis1_start = origin
#     axis1_end = origin + principal_axes[0] * max_distance
#     fig.add_trace(go.Scatter3d(x=[axis1_start[0], axis1_end[0]], y=[axis1_start[1], axis1_end[1]], z=[axis1_start[2], axis1_end[2]], mode='lines', line=dict(color='red')))

#     # Plot the second principal axis
#     axis2_start = origin
#     axis2_end = origin + principal_axes[1] * max_distance
#     fig.add_trace(go.Scatter3d(x=[axis2_start[0], axis2_end[0]], y=[axis2_start[1], axis2_end[1]], z=[axis2_start[2], axis2_end[2]], mode='lines', line=dict(color='green')))

#     # Plot the third principal axis
#     axis3_start = origin
#     axis3_end = origin +  principal_axes[2] * max_distance
#     fig.add_trace(go.Scatter3d(x=[axis3_start[0], axis3_end[0]], y=[axis3_start[1], axis3_end[1]], z=[axis3_start[2], axis3_end[2]], mode='lines', line=dict(color='blue')))

#     # Set plot layout
#     fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

#     # Show the plot
#     fig.show()

def get_vertices_vtk(f_path, outpath):
    # Load the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(f_path))
    reader.Update()

    # Get the output polydata
    polydata = reader.GetOutput()

    # Get the points (vertices) from the polydata
    points = polydata.GetPoints()

    # Open a text file to write the vertices
    with open(outpath, 'w') as file:
        # Write each vertex as a line in the text file
        for i in range(points.GetNumberOfPoints()):
            vertex = points.GetPoint(i)
            file.write(f'{vertex[0]} {vertex[1]} {vertex[2]}\n')

    print('Extraction complete.')

def align_vtk_to_origin(f_path, output):

    # Load the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(f_path))
    reader.Update()

    # Get the output polydata
    polydata = reader.GetOutput()

    # Get the points (vertices) from the polydata
    points = polydata.GetPoints()

    # Compute the centroid
    centroid = [0.0, 0.0, 0.0]
    num_points = points.GetNumberOfPoints()
    for i in range(num_points):
        point = points.GetPoint(i)
        centroid = [c + p for c, p in zip(centroid, point)]
    centroid = [c / num_points for c in centroid]

    # Subtract the centroid from each point
    for i in range(num_points):
        point = points.GetPoint(i)
        aligned_point = [p - c for p, c in zip(point, centroid)]
        points.SetPoint(i, aligned_point)

    # Save the aligned points to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(polydata)
    writer.Write()

    return centroid

def get_principal_axes(vtk_file, plot=False):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_file))
    reader.Update()

    # Get the vertex positions from the VTK dataset
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    #also get the faces
    faces = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        face = []
        for j in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(j)
            face.append(point_id)
        faces.append(face)

    # Step 2: Compute the centroid
    centroid = np.mean(vertices, axis=0)

    # Step 3: Center the data
    centered_vertices = vertices - centroid

    # Step 4: Construct the data matrix
    data_matrix = centered_vertices

    # Step 5: Perform PCA
    covariance_matrix = np.cov(data_matrix, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 6: Extract the principal axes
    principal_axes = eigenvectors.T

    print(principal_axes)

    #plot the eigenvalues and the mesh
    if plot:
        plot_mesh_and_pricipal_axes(vertices, centroid, eigenvalues, principal_axes)

    return vertices, faces, principal_axes, eigenvalues

def get_points_vtk(vtk_file):
    # Load the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_file))
    reader.Update()

    # Get the polydata object from the reader
    polydata = reader.GetOutput()

    # Get the points from the polydata
    points = polydata.GetPoints()

    # Get the number of points
    num_points = points.GetNumberOfPoints()

    point_list = []
    for i in range(num_points):
        point = points.GetPoint(i)
        point_list.append(point)

    return polydata, point_list, num_points

def apply_rotation_vtk(vtk_file, rot, output):
    # Load the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_file))
    reader.Update()

    # Get the polydata object from the reader
    polydata = reader.GetOutput()

    # Get the points from the polydata
    points = polydata.GetPoints()

    # Get the number of points
    num_points = points.GetNumberOfPoints()

    # Create a new array to store the rotated points
    rotated_points = vtk.vtkPoints()
    rotated_points.SetNumberOfPoints(num_points)

    # Apply the rotation matrix to each point
    for i in range(num_points):
        point = points.GetPoint(i)
        rotated_point = np.dot(rot, point)
        rotated_points.SetPoint(i, rotated_point)

    # Set the rotated points back to the polydata
    polydata.SetPoints(rotated_points)

    # Save the modified polydata to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(polydata)
    writer.Write()

def icp_registration(static_vtk, moving_vtk):
    
    spolydata, spoints, snum_points = get_points_vtk(static_vtk)
    mpolydata, mpoints, mnum_points = get_points_vtk(moving_vtk)

    mtensor = torch.tensor(mpoints).unsqueeze(0)
    mtensor.to('cpu')
    stensor = torch.tensor(spoints).unsqueeze(0)
    stensor.to('cpu')
    ICP_sol = iterative_closest_point(mtensor, stensor)

    return ICP_sol, mpolydata

def output_new_vtk(polydata, newpoints, outfile):

    # Get the points from the polydata
    points = polydata.GetPoints()

    # Get the number of points
    num_points = points.GetNumberOfPoints()

    # Create a new array to store the new points
    new_points = vtk.vtkPoints()
    new_points.SetNumberOfPoints(num_points)

    # Apply the rotation matrix to each point
    for i in range(num_points):
        point = newpoints[i,:]
        new_points.SetPoint(i, point)

    # Set the rotated points back to the polydata
    polydata.SetPoints(new_points)

    #output the new vtk file
        # Save the modified polydata to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(outfile))
    writer.SetInputData(polydata)
    writer.Write()





###########################
## Main function ##########
###########################

def register_vtk_to_nifti(nifti, obj, png, output_dir, threshold_value=200, inversions=[False, False, False]):

    #first, convert the obj file to vtk
    print("***CONVERTING OBJ TO VTK***")
    vtk_from_obj = output_dir/("moving_mesh.vtk")
    utils.create_vtk_from_obj(obj, png, vtk_from_obj)
    print("***DONE CONVERTING OBJ TO VTK***")

    #also convert the CT nifti into a binary mask that we will use to create a surface
    print("***CONVERTING CT TO VTK***")
    mask = output_dir/("mask.nii.gz")
    bodyFatSofttissueMask.mask_CT(nifti, mask, threshold_value=threshold_value)
        ###NOTE: nii2mesh cannot work on an 8bit int image or 64 float (look in source code to see which ones work)
        ###Make sure that the mask coming out of the body tissue mask is 16 bit int
    static_vtk = output_dir/("static.vtk")
    nii2mesh_cmd = "nii2mesh {} -v 1 {}".format(str(mask), str(static_vtk))
    subprocess.run(nii2mesh_cmd, shell=True)
    #subprocess.run(['nii2mesh', '{}', '-v', '1', '{}'.format(str(mask), str(static_vtk))])
    print("***DONE CONVERTING CT TO VTK***")

    ## Next, align the input vtk meshes centroids to the origin
    print("***ALIGNING MESHES TO ORIGIN***")
    static_vtk_centered = output_dir/("static_centered.vtk")
    moving_vtk_centered = output_dir/("moving_centered.vtk")
    align_vtk_to_origin(static_vtk, static_vtk_centered)
    align_vtk_to_origin(vtk_from_obj, moving_vtk_centered)
    print("***DONE ALIGNING MESHES TO ORIGIN***")
    #align_vtk_to_origin('/home-local/kimm58/SPIE2023/mask_out/output_mesh.vtk', '/home-local/kimm58/SPIE2023/mask_out/output_mesh_centered.vtk')
    #align_vtk_to_origin('/home-local/kimm58/SPIE2023/data/H1Capture/hip_on_bottle.vtk', '/home-local/kimm58/SPIE2023/data/H1Capture/hip_on_bottle_centered.vtk')


    ## Then, do a preliminary alignment based on PCA (specify which axes to invert for PCA)
    #get the principal axes and pointclouds
    print("***BEGINNING PCA ALIGNMENT***")
    svertices, sfaces, sprincipal_axes, seigenvalues = get_principal_axes(static_vtk_centered, plot=False)
    mvertices, mfaces, mprincipal_axes, meigenvalues = get_principal_axes(moving_vtk_centered, plot=False)

    ### Need to specify the inversion of axes
    for i,invert in enumerate(inversions):
        if invert:
            mprincipal_axes[i] = -mprincipal_axes[i]
    #mprincipal_axes[1] = -mprincipal_axes[1]

    #calculate the rotation from PCA
    rot = np.dot(sprincipal_axes.T, mprincipal_axes)
    pca_rot = output_dir/("PCA_rotation.npy")
    print("***CALCULATED PCA REGISTRATION. SAVING REGISTRATION TO {}***".format(str(pca_rot)))
    np.savetxt(pca_rot, rot)

    #apply the rotation to the moving vtk file and save it
    print("***APPLYING PCA ALIGNMENT***")
    pca_aligned = output_dir/("moving_PCA_aligned.vtk")
    apply_rotation_vtk(moving_vtk_centered, rot, pca_aligned)
    print("***DONE APPLYING PCA ALIGNMENT***")

    #now that they are principally aligned, use ICP to do a better alignment
    print("***COMPUTING ICP ALIGNMENT***")
    ICP_sol, mpolydata = icp_registration(static_vtk_centered, pca_aligned)

    #get the registered points and transforms
    moved_points = ICP_sol.Xt.squeeze(0).numpy()
    icp_r = ICP_sol.RTs.R.squeeze(0).numpy()
    icp_t = ICP_sol.RTs.T.squeeze(0).numpy()
    icp_s = ICP_sol.RTs.s.numpy()


    #output the registered surface to a new vtk file
    print("***DONE COMPUTING ICP ALIGNMENT. SAVING REGISTERED VTK FILE AND TRANSFORMS***")
        #save new vtk file
    ICP_aligned_vtk = output_dir/("moving_ICP_aligned.vtk")
    output_new_vtk(mpolydata, moved_points, ICP_aligned_vtk)
        #save icp transofrms
    icp_r_f = output_dir/("ICP_rotation.npy")
    icp_r_t = output_dir/("ICP_translation.npy")
    icp_r_s = output_dir/("ICP_scale.npy")
    np.savetxt(icp_r_f, icp_r)
    np.savetxt(icp_r_t, icp_t)
    np.savetxt(icp_r_s, icp_s)






#align the surfaces to the origin
# CTcentroid = align_vtk_to_origin('/home-local/kimm58/SPIE2023/mask_out/output_mesh.vtk', '/home-local/kimm58/SPIE2023/mask_out/output_mesh_centered.vtk')
# H1centroid = align_vtk_to_origin('/home-local/kimm58/SPIE2023/data/H1Capture/hip_on_bottle.vtk', '/home-local/kimm58/SPIE2023/data/H1Capture/hip_on_bottle_centered.vtk')

# static_vtk = '/home-local/kimm58/SPIE2023/data/regTest/ICP_PCA/output_mesh_centered.vtk'
# moving_vtk = '/home-local/kimm58/SPIE2023/data/regTest/ICP_PCA/hip_on_bottle_centered.vtk'
# output_vtk = '/home-local/kimm58/SPIE2023/data/regTest/ICP_PCA/hip_on_bottle_registered_temp.vtk'
# final_output_vtk = '/home-local/kimm58/SPIE2023/data/regTest/ICP_PCA/hip_on_bottle_registered.vtk'


#get the principal axes and pointclouds
# svertices, sfaces, sprincipal_axes, seigenvalues = get_principal_axes(static_vtk, plot=False)
# mvertices, mfaces, mprincipal_axes, meigenvalues = get_principal_axes(moving_vtk, plot=False)

#this is required to invert the object if it needs to be
    #no idea how to check if this is necessary though
# mprincipal_axes[1] = -mprincipal_axes[1]

#determine the rotation to align the axes
#rot = np.dot(sprincipal_axes.T, mprincipal_axes)

#apply the rotation to the moving vtk file and save it
#apply_rotation_vtk(moving_vtk, rot, output_vtk)

#now that they are principally aligned, use ICP to do a better alignment
#ICP_sol, mpolydata = icp_registration(static_vtk, output_vtk, final_output_vtk)

#get the registered points
#moved_points = ICP_sol.Xt.squeeze(0).numpy()

#output the registered surface to a new vtk file
#output_new_vtk(mpolydata, moved_points)