from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import torch
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds
import vtk
import trimesh
import numpy as np
import subprocess
import pandas as pd
from tqdm import tqdm

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

def calc_fiducial_mse(v1, v2):
    #given two sets of vertices, calculate the MSE distance between them
    #assumes that they are ordered
        #otherwise, finds the closest pairs?
    
    dist_sum = 0
    for i in range(v1.shape[0]):
        dist_sum += np.linalg.norm(v1[i], v2[i])
    
    print("Dist sum:", dist_sum)
    return dist_sum

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

    faces = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        face = []
        for j in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(j)
            face.append(point_id)
        faces.append(face)

    return polydata, point_list, num_points, faces

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

def icp_registration(static_vtk, moving_vtk, estimate_scale=False):
    
    spolydata, spoints, snum_points, sfaces = get_points_vtk(static_vtk)
    mpolydata, mpoints, mnum_points, mfaces = get_points_vtk(moving_vtk)

    mtensor = torch.tensor(mpoints).unsqueeze(0)
    mtensor.to('cpu')
    stensor = torch.tensor(spoints).unsqueeze(0)
    stensor.to('cpu')
    ICP_sol = iterative_closest_point(mtensor, stensor, estimate_scale=estimate_scale)

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

#reads a csv to get the fiducials
    #assumes that the points were selected via paraview and exported as a csv
def get_fiducials_from_csv(path):

    df = pd.read_csv(path)

    points = []
    for row in df.iterrows():
        
        x = row[1]['Points_0']
        y = row[1]['Points_1']
        z = row[1]['Points_2']
        points.append(np.array([x,y,z]))

    return np.stack(points)

#applies the transforms to the fiducials
def apply_transforms_to_fiducials(PCA_r, ICP_r, ICP_t, points, debug=False):
    #points is an Nx3 numpy array

    #rot1
    if debug:
        print("Original Centered")
        print(points)
    #x = np.dot(points, PCA_r)
    x = np.dot(points, PCA_r.T)
    if debug:
        print("PCA aligned")
        print(x)
    x = np.dot(x, ICP_r)
    if debug:
        print("ICP rotation")
        print(x)
    x = x + ICP_t
    if debug:
        print("ICP translation")
        print(x)

    return x

#get the transforms
def get_transforms(rootpath):

    #gets the transforms for PCA and ICP
    outputs = rootpath/("outputs")
    pca_r = np.loadtxt(outputs/("PCA_rotation.npy"))
    icp_r = np.loadtxt(outputs/("ICP_rotation.npy"))
    icp_t = np.loadtxt(outputs/("ICP_translation.npy"))

    return (pca_r, icp_r, icp_t)



#scale up the surface to the proper dimensions
def scale_up_mesh(static, moving, output):

    def get_furthest_distance(points):
        longest = 0
        pair = ()
        numpoints = points.GetNumberOfPoints()
        for i in tqdm(range(numpoints)):
            point1 = points.GetPoint(i)
            for j in range(i+1, numpoints):
                point2 = points.GetPoint(j)
                dist = vtk.vtkMath.Distance2BetweenPoints(point1, point2)

                if dist > longest:
                    longest = dist
                    pair = (point1, point2)
    
        return pair, longest

    #read in the polydatas    
    static_reader = vtk.vtkPolyDataReader()
    moving_reader = vtk.vtkPolyDataReader()
    static_reader.SetFileName(str(static))
    moving_reader.SetFileName(str(moving))
    static_reader.Update()
    moving_reader.Update()

    #get the furthest distance between any two points for the static data
    print("Calculating the furthest distance between any two points on the static mesh...")
    static_pair, static_dist = utils.appx_max_distance_mesh(static_reader.GetOutput().GetPoints()) #get_furthest_distance(static_reader.GetOutput().GetPoints())

    #do the same for the moving mesh
    moving_poly = moving_reader.GetOutput()
    moving_points = moving_poly.GetPoints()
    print("Calculating the furthest distance between any two points on the moving mesh...")
    moving_pair, moving_dist = utils.appx_max_distance_mesh(moving_points)

    #now, find the scaling factor and use it to scale up the points
    scale = static_dist / moving_dist
    print(scale)
    print(static_dist)
    print(moving_dist)
    print("static pair:",static_pair)
    print("moving pair:",moving_pair)

    #apply the transform to the moving mesh
    print("Applying the scaling to the moving mesh...")
    for i in tqdm(range(moving_points.GetNumberOfPoints())):
        point = moving_points.GetPoint(i)
        expanded_point = [coord * scale for coord in point]
        moving_points.SetPoint(i, expanded_point)


    #now save the output of the scaled mesh
    print("Now saving the scaled mesh...")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(moving_poly)
    writer.Write()

    return scale

def create_vtk_pointcloud(array, output):
    #given a numpy array of size Nx3, outputs a pointcloud
    polydata = vtk.vtkPolyData()

    vtk_points = vtk.vtkPoints()
    for point in array:
        vtk_points.InsertNextPoint(point)
    polydata.SetPoints(vtk_points)

    # Create a vtkVertexGlyphFilter to convert points to vertices
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    point_cloud_polydata = vertex_filter.GetOutput()

    # Save the point cloud as a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(point_cloud_polydata)
    writer.Write()

#performs a reverse ICP registration, then inverts the transform and applies it to the NeRF data
    #helps remove registration error due to noise points in the NeRF
def reverse_icp_registration(CT, nerf, output, R=None, t=None, s=None):
    #given the transforms and the scale, reverse the ICP transform so that the nerf surface moves to the CT
    #read in the polydatas
    ct_poly = utils.get_polydata_vtk(CT)
    nerf_poly = utils.get_polydata_vtk(nerf)

    if R == None and s == None and t == None:
        print("Performing ICP registration from CT to NeRF surface...")
        ICP_sol, moved_polydata = icp_registration(nerf, CT, estimate_scale=True)
        R = ICP_sol.RTs.R
        t = ICP_sol.RTs.T
        s = ICP_sol.RTs.s


    nerf_points = nerf_poly.GetPoints()

    #get the inverse transforms
    rinv = np.linalg.inv(R.squeeze(0).numpy())
    tinv = -t.numpy()
    sinv = 1/s.numpy()

    #apply the inverse rotation, translation, and scaling
    print("Applying reverse registration and scaling to the points...")
    for i in tqdm(range(nerf_points.GetNumberOfPoints())):
        point = nerf_points.GetPoint(i)
        shifted_point = point + tinv
        rotated_point = np.dot(shifted_point, rinv)
        expanded_point = [coord * sinv for coord in rotated_point]
        nerf_points.SetPoint(i, expanded_point[0])

    #save the output
    print("***DONE COMPUTING ICP ALIGNMENT. SAVING REGISTERED VTK FILE***")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(nerf_poly)
    writer.Write()

    return rinv, tinv, sinv

    #given two sets of points, plots the fiducials

###########################
## Main function ##########
###########################

def register_vtk_to_nifti(nifti, src, png, output_dir, threshold_value=200, inversions=[False, False, False]):

    #first, convert the obj file to vtk

    if str(src).endswith('.obj'):
        print("***CONVERTING OBJ TO VTK***")
        vtk_from_obj = output_dir/("moving_mesh.vtk")
        utils.create_vtk_from_obj(src, png, vtk_from_obj)
        print("***DONE CONVERTING OBJ TO VTK***")
    elif str(src).endswith('.vtk'):
        vtk_from_obj = src

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
    print("{} and {}".format(str(static_vtk), str(vtk_from_obj)))
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
    ICP_aligned_vtk = output_dir/("moving_ICP_aligned.vtk")
    if str(src).endswith('.obj'):
        ICP_sol, mpolydata = icp_registration(static_vtk_centered, pca_aligned)
        #get the registered points and transforms
        moved_points = ICP_sol.Xt.squeeze(0).numpy()
        icp_r = ICP_sol.RTs.R.squeeze(0).numpy()
        icp_t = ICP_sol.RTs.T.squeeze(0).numpy()
        icp_s = ICP_sol.RTs.s.numpy()
            #output the registered surface to a new vtk file
        print("***DONE COMPUTING ICP ALIGNMENT. SAVING REGISTERED VTK FILE***")
            #save new vtk file
        output_new_vtk(mpolydata, moved_points, ICP_aligned_vtk)
    #for NeRF registration
        #need to do a preliminary scaling to make the NeRF surface roughly the same size as the CT
        #need to do a reverse ICP, then apply the inverted transforms
        #saves the inverted transforms (from NeRF to CT)
    else:
        #preliminary scaling
        print("Data from NeRF, not H1 Camera. Using scale for ICP registration...")
        print("Scaling the NeRF surface to be approximately the same size as the CT surface...")
        moving_scaled = output_dir/("moving_scaled.vtk")
        prelim_scale = scale_up_mesh(static_vtk_centered, pca_aligned, moving_scaled)
        print("Done with preliminary scaling. Saving the preliminary scale factor...")
        prelim_scale_f = output_dir/("prelim_scale.npy")
        np.savetxt(prelim_scale_f, np.array([prelim_scale]))

        print("Now registering CT to scaled NeRF via ICP. Using inverse transforms to move NeRF to CT space...")
        #ICP_sol, mpolydata = icp_registration(static_vtk_centered, pca_aligned, estimate_scale=True)
        icp_r, icp_t, icp_s = reverse_icp_registration(static_vtk_centered, moving_scaled, ICP_aligned_vtk, R=None, t=None, s=None)


    #save icp transofrms
    print("***SAVING TRANSFORMS***")
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