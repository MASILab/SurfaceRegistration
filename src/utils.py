import operator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import vtk
from tqdm import tqdm
import trimesh
from vtk.util.numpy_support import numpy_to_vtk
import nibabel as nib
import pandas as pd
from scipy.spatial import ConvexHull

import registration

###
### Convert OBJ and PNG to VTK
###
def get_color_from_uv_coord(uv, image, test=False):
    width, height = image.size

    pix_x, pix_y = uv[0] * width, (1 - uv[1]) * height
    pix_x = max(0, min(pix_x, width - 1))
    pix_y = max(0, min(pix_y, height - 1))

    if test:
        print(width, height)
        print(pix_x, pix_y)
        plt.imshow(image)
        plt.plot(pix_x, pix_y, 'ro')

    color = image.getpixel((pix_x, pix_y))
    return color

def create_vtk_from_obj(obj, texture_file, output):
    #get the vertices, colors, normals, and faces
    vertices, normals, colors, faces = parse_obj(obj, texture_file)

    #print(faces)

    #use these to create a vtk object

    # Create VTK objects
    poly_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    vertex_colors = vtk.vtkUnsignedCharArray()
    vertex_normals = vtk.vtkFloatArray()
    triangles = vtk.vtkCellArray()

    # Set vertices
    for vertex in vertices:
        points.InsertNextPoint(vertex)
    poly_data.SetPoints(points)

    # Set vertex colors
    vertex_colors.SetNumberOfComponents(3)  # RGB values
    for color in colors:
        vertex_colors.InsertNextTuple3(*color)
    poly_data.GetPointData().SetScalars(vertex_colors)

    # Set vertex normals
    # vertex_normals.SetNumberOfComponents(3)  # XYZ values
    # for normal in normals:
    #     vertex_normals.InsertNextTuple3(*normal)
    # poly_data.GetPointData().SetNormals(vertex_normals)

    #Add the normals to the VTK data as well
    vertex_normals.SetName("Normals")
    vertex_normals.SetNumberOfComponents(3)

    for normal in normals:
        vertex_normals.InsertNextTuple3(normal[0], normal[1], normal[2])

    poly_data.GetPointData().SetNormals(vertex_normals)

    # Set faces
    for face in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, face[0])
        triangle.GetPointIds().SetId(1, face[1])
        triangle.GetPointIds().SetId(2, face[2])
        triangles.InsertNextCell(triangle)
    poly_data.SetPolys(triangles)

    # Write the VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(poly_data)
    writer.Write()

def parse_obj(obj, texture_file):

    #open the texture image png/jpeg
    texture_image = Image.open(texture_file)

    #read in the obj file
    with open(obj, 'r') as file:
        vertices = []
        normals = []
        colors = []
        faces = []
        for line in file:
            line = line.strip()
            if line:
                values = line.split()
                category = values[0]
                if category == 'mtllib' or category == 'usemtl':
                    continue
                elif category == 'v':
                    coords = np.array(values[1:]).astype(float)
                    vertices.append(coords)
                elif category == 'vn':
                    coords = np.array(values[1:]).astype(float)
                    normals.append(coords)
                elif category == 'vt':
                    pix_coords = np.array(values[1:]).astype(float)
                    color = get_color_from_uv_coord(pix_coords, texture_image)
                    colors.append(color)
                elif category == 'f':
                    #parse these later
                    faces.append(values[1:])
            #break
        #print(faces)
    
    #now, turn each into a numpy array
    #vertices = np.stack(vertices)
    #normals = np.stack(normals)
    #colors = np.stack(colors)

    #get a list of the unique vertex information from the faces
        # we want to map each vertex to its correct normal and color values
    uniq = []
    vertex_unordered = []
    normal_unordered = []
    color_unordered = []
    coord_idxs = []
    u = 0
    for face in faces:
        for vertex in face:
            coord_idx, texture_idx, norm_idx = vertex.split('/')
            if coord_idx in uniq:
                continue
            uniq.append(coord_idx)
            cs_coord_idx = int(coord_idx)-1
            vertex_unordered.append((vertices[cs_coord_idx], cs_coord_idx))
            normal_unordered.append((normals[int(norm_idx)-1], cs_coord_idx))
            color_unordered.append((colors[int(texture_idx)-1], cs_coord_idx))
            #coord_idxs.append(int(coord_idx)-1)

    #have to order these by coord index
    vertex_ordered = [x[0] for x in sorted(vertex_unordered, key=lambda x: x[1])]
    normal_ordered = [x[0] for x in sorted(normal_unordered, key=lambda x: x[1])]
    color_ordered = [x[0] for x in sorted(color_unordered, key=lambda x: x[1])]

            #check to see the vertex color and position
            # if coord_idx == '2139':
            #     print("colors:", colors[int(texture_idx)-1])
            #     print("coords:", vertices[int(coord_idx)-1])
            

    #now, parse each of the faces
    final_faces = []
    for face in faces:
        final_face = []
        for vertex in face:
            coord_idx, texture_idx, norm_idx = vertex.split('/')
            #get the proper vertex
            final_face.append(int(coord_idx)-1)
        final_faces.append(final_face)

    #now that we have the vertices, faces, colors, and normals, turn this into a vtk file
    return vertex_ordered, normal_ordered, color_ordered, final_faces


### Convert btw file types
def convert_stl_to_vtk(stl_file_path, outpath):

    # Read the STL file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_file_path))
    reader.Update()

    # Get the polydata from the reader
    polydata = reader.GetOutput()

    # Write the polydata to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(outpath)
    writer.SetInputData(polydata)
    writer.Write()

###
### NeRF Pipeline
###
#scales up the vertices
def scale_up_vertices_obj(input, output):
    # Create the PLY reader
    reader = vtk.vtkOBJReader()
    reader.SetFileName(input)

    # Read the PLY file
    reader.Update()

    # Get the output polydata
    polydata = reader.GetOutput()

    # Access the vertices in the polydata
    points = polydata.GetPoints()

    # Define the scaling factor for expansion
    scaling_factor = 40.0

    # Expand the vertices by scaling their coordinates
    for i in tqdm(range(points.GetNumberOfPoints())):
        point = points.GetPoint(i)
        expanded_point = [coord * scaling_factor for coord in point]
        points.SetPoint(i, expanded_point)

    # Optionally, you can write the modified polydata to a new PLY file
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(polydata)
    writer.Write()


#remove colors from the vertices
    #if green is higher than 200
#def remove_colors_from_mesh(mesh, output):
    face_colors = mesh.visual.face_colors[:, :3]
    color_threshold = [0,200,0]
    faces_to_keep = np.any(face_colors <= color_threshold, axis=1)

    selectedv = mesh.vertices
    selectedf = mesh.faces[faces_to_keep]

    # Create a new mesh with the selected vertices and faces
    new_mesh = trimesh.Trimesh(vertices=selectedv, faces=selectedf)

    # Remove isolated vertices
    new_mesh.remove_unreferenced_vertices()

    new_mesh.export(output, file_type='ply')

def position_threshold_mesh(input, output, min=[-0.4, -0.4, -0.4], max=[0.4, 0.4, 0.4]):
    #remove all vertices that do not fall within the box defined by min and max

    def is_outside_bounds(vertex):
        return 

    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()

    # Get the polydata from the reader
    polydata = reader.GetOutput()

    # Get the point data and RGB color array
    point_data = polydata.GetPointData()
    color_array = point_data.GetArray("Colors")

    # Create a mask for the vertices to remove
    mask = np.ones(polydata.GetNumberOfPoints(), dtype=bool)

    # Iterate over the points
    for i in range(polydata.GetNumberOfPoints()):
        
        vertex = np.array(polydata.GetPoint(i))
        if np.any(vertex < min) or np.any(vertex > max):
            mask[i] = False #mark vertex for removal

        # exit(0)
        # ####
        # if vertex:
        #     mask[i] = False  # Mark the vertex for removal

    print("Done creating mask. Now using mask to filter out vertices...")
    # Create a new polydata to hold the filtered data
    filtered_polydata = vtk.vtkPolyData()

    # Copy the remaining points to the filtered polydata
    points = vtk.vtkPoints()
    filtered_colors = vtk.vtkUnsignedCharArray()
    filtered_colors.SetNumberOfComponents(3)
    filtered_colors.SetName("Colors")

    # Create a map from old vertex IDs to new vertex IDs
    vertex_map = {}

    print("Total points:", polydata.GetNumberOfPoints())
    print("Number of points to keep:", np.count_nonzero(mask))

    for i in range(polydata.GetNumberOfPoints()):
        if mask[i]:
            new_vertex_id = points.InsertNextPoint(polydata.GetPoint(i))
            #points.InsertNextPoint(polydata.GetPoint(i))
            vertex_map[i] = new_vertex_id
            color_tuple = color_array.GetTuple(i)
            #print(i)
            #print(polydata.GetPoint(i))
            #print(color_tuple)
            filtered_colors.InsertNextTuple3(color_tuple[0], color_tuple[1], color_tuple[2])
            #print(filtered_colors.GetTuple(filtered_colors.GetNumberOfTuples() - 1))
            #exit(0)
            #vtk_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))

    filtered_polydata.SetPoints(points)
    filtered_polydata.GetPointData().SetScalars(filtered_colors)

    print("Now removing faces associated with removed vertices...")

    # Remove the faces associated with the removed vertices
    filtered_faces = vtk.vtkCellArray()
    faces = polydata.GetPolys()
    faces.InitTraversal()

    while True:
        face = vtk.vtkIdList()
        if faces.GetNextCell(face) == 0:
            break
        # Check if all face vertices are in the vertex map
        if all(vertex_map.get(face.GetId(j)) is not None for j in range(3)):
            filtered_face = vtk.vtkIdList()
            for j in range(3):
                vertex_id = vertex_map[face.GetId(j)]
                filtered_face.InsertNextId(vertex_id)
            filtered_faces.InsertNextCell(filtered_face)

    print("Done filtering faces. Now exporting vtk file...")

    filtered_polydata.SetPolys(filtered_faces)

    # Write the filtered polydata to a new VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(filtered_polydata)
    writer.Write()

#thresholds a mesh based on color
    #tailored to green for NERF
def color_threshold_mesh(input, output, abs_threshold=[200,200,200], ratio_threshold=0.75, g_abs_thresh=90, H1=False):

    def pass_abs_threshold(color, R, G, B):
        return color[0] < R and color[1] > G and color[2] < B
    
    def pass_ratio_threshold(color):
        red, green, blue = color[0], color[1], color[2]
        if green == 0:
            return False
        return (red/green < ratio_threshold and blue/green < ratio_threshold and green > g_abs_thresh) #or red/green < 0.85 and blue/green < 0.85

    def H1_pass_abs_threshold(color, R, G, B):
        return color[0] < R and color[1] > G and color[2] > B
    
    def H1_pass_ratio_threshold(color):
        red, green, blue = color[0], color[1], color[2]
        if blue == 0 or green == 0:
            return False
        return red/green < ratio_threshold and red/blue < ratio_threshold and green > g_abs_thresh

    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()

    # Get the polydata from the reader
    polydata = reader.GetOutput()

    # Get the point data and RGB color array
    point_data = polydata.GetPointData()
    if not H1:
        color_array = point_data.GetArray("Colors")
    else:
        color_array = point_data.GetScalars()

    # Create a mask for the vertices to remove
    mask = np.ones(polydata.GetNumberOfPoints(), dtype=bool)

    # Iterate over the points and colors
    R = abs_threshold[0]
    G = abs_threshold[1]
    B = abs_threshold[2]
    print("Creating mask based on color threshold of [{},{},{}]...".format(R, G, B))
    #NERF
    if not H1:
        for i in range(polydata.GetNumberOfPoints()):
            color = color_array.GetTuple(i)

            if pass_abs_threshold(color, R, G, B) or pass_ratio_threshold(color):
                mask[i] = False  # Mark the vertex for removal
    #H1 camera segmentation: blue background
    else:
        #over 190 BG, R < 110
        #also 60% ratio threshold
        for i in range(polydata.GetNumberOfPoints()):
            color = color_array.GetTuple(i)

            if H1_pass_abs_threshold(color, R, G, B) or H1_pass_ratio_threshold(color):
                mask[i] = False  # Mark the vertex for removal

    print("Done creating mask. Now using mask to filter out vertices...")
    # Create a new polydata to hold the filtered data
    filtered_polydata = vtk.vtkPolyData()

    # Copy the remaining points to the filtered polydata
    points = vtk.vtkPoints()
    filtered_colors = vtk.vtkUnsignedCharArray()
    filtered_colors.SetNumberOfComponents(3)
    filtered_colors.SetName("Colors")

    # Create a map from old vertex IDs to new vertex IDs
    vertex_map = {}

    print("Total points:", polydata.GetNumberOfPoints())
    print("Number of points to keep:", np.count_nonzero(mask))

    for i in range(polydata.GetNumberOfPoints()):
        if mask[i]:
            new_vertex_id = points.InsertNextPoint(polydata.GetPoint(i))
            #points.InsertNextPoint(polydata.GetPoint(i))
            vertex_map[i] = new_vertex_id
            color_tuple = color_array.GetTuple(i)
            #print(i)
            #print(polydata.GetPoint(i))
            #print(color_tuple)
            filtered_colors.InsertNextTuple3(color_tuple[0], color_tuple[1], color_tuple[2])
            #print(filtered_colors.GetTuple(filtered_colors.GetNumberOfTuples() - 1))
            #exit(0)
            #vtk_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))

    filtered_polydata.SetPoints(points)
    filtered_polydata.GetPointData().SetScalars(filtered_colors)

    print("Now removing faces associated with removed vertices...")

    # Remove the faces associated with the removed vertices
    filtered_faces = vtk.vtkCellArray()
    faces = polydata.GetPolys()
    faces.InitTraversal()

    while True:
        face = vtk.vtkIdList()
        if faces.GetNextCell(face) == 0:
            break
        # Check if all face vertices are in the vertex map
        if all(vertex_map.get(face.GetId(j)) is not None for j in range(3)):
            filtered_face = vtk.vtkIdList()
            for j in range(3):
                vertex_id = vertex_map[face.GetId(j)]
                filtered_face.InsertNextId(vertex_id)
            filtered_faces.InsertNextCell(filtered_face)

    print("Done filtering faces. Now exporting vtk file...")

    filtered_polydata.SetPolys(filtered_faces)

    # Write the filtered polydata to a new VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(filtered_polydata)
    writer.Write()


def remove_paraview_selected_points(input, csv, outfile, H1=False):

    def get_points_list(csv):
        df = pd.read_csv(csv)
        return df['Point ID'].tolist()

    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()

    # Get the polydata from the reader
    polydata = reader.GetOutput()

    # Get the point data and RGB color array
    point_data = polydata.GetPointData()
    if not H1:
        color_array = point_data.GetArray("Colors")
    else:
        color_array = point_data.GetScalars()

    # Create a mask for the vertices to remove
    mask = np.ones(polydata.GetNumberOfPoints(), dtype=bool)

    #get the list of point IDs to remove
    delete_points = get_points_list(csv)

    # Iterate over the points and colors
    print("Removing selected points...")

    for i in range(polydata.GetNumberOfPoints()):

        if i in delete_points:
            mask[i] = False  # Mark the vertex for removal

    print("Done creating mask. Now using mask to filter out vertices...")
    # Create a new polydata to hold the filtered data
    filtered_polydata = vtk.vtkPolyData()

    # Copy the remaining points to the filtered polydata
    points = vtk.vtkPoints()
    filtered_colors = vtk.vtkUnsignedCharArray()
    filtered_colors.SetNumberOfComponents(3)
    filtered_colors.SetName("Colors")

    # Create a map from old vertex IDs to new vertex IDs
    vertex_map = {}

    print("Total points:", polydata.GetNumberOfPoints())
    print("Number of points to keep:", np.count_nonzero(mask))

    for i in range(polydata.GetNumberOfPoints()):
        if mask[i]:
            new_vertex_id = points.InsertNextPoint(polydata.GetPoint(i))
            vertex_map[i] = new_vertex_id
            color_tuple = color_array.GetTuple(i)

            filtered_colors.InsertNextTuple3(color_tuple[0], color_tuple[1], color_tuple[2])

    filtered_polydata.SetPoints(points)
    filtered_polydata.GetPointData().SetScalars(filtered_colors)

    print("Now removing faces associated with removed vertices...")

    # Remove the faces associated with the removed vertices
    filtered_faces = vtk.vtkCellArray()
    faces = polydata.GetPolys()
    faces.InitTraversal()

    while True:
        face = vtk.vtkIdList()
        if faces.GetNextCell(face) == 0:
            break
        # Check if all face vertices are in the vertex map
        if all(vertex_map.get(face.GetId(j)) is not None for j in range(3)):
            filtered_face = vtk.vtkIdList()
            for j in range(3):
                vertex_id = vertex_map[face.GetId(j)]
                filtered_face.InsertNextId(vertex_id)
            filtered_faces.InsertNextCell(filtered_face)

    print("Done filtering faces. Now exporting vtk file...")

    filtered_polydata.SetPolys(filtered_faces)

    # Write the filtered polydata to a new VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(outfile))
    writer.SetInputData(filtered_polydata)
    writer.Write()

def get_largest_connected_component(input, output):

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()

    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(reader.GetOutput())
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.Update()

    outvtk = connectivity_filter.GetOutput()

    print(connectivity_filter.GetNumberOfExtractedRegions())

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(outvtk)
    writer.Write()

def get_largest_component_near_origin(input, output, radius=1.0):
    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()

    # Create a sphere centered at the origin
    sphere = vtk.vtkSphere()
    sphere.SetCenter(0, 0, 0)
    sphere.SetRadius(radius)  # Set the radius of the ball

    # Apply vtkIntersectionPolyDataFilter to extract the intersection with the sphere
    intersection_filter = vtk.vtkIntersectionPolyDataFilter()
    intersection_filter.SetInputData(0, reader.GetOutput())
    intersection_filter.SetGeometry(1, sphere)
    intersection_filter.Update()

    # Apply the vtkConnectivityFilter to extract the largest connected component
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputConnection(intersection_filter.GetOutputPort())
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.ColorRegionsOn()  # Optional: Color regions for visualization
    connectivity_filter.Update()

    # Get the output of the connectivity filter
    outvtk = connectivity_filter.GetOutput()

    # Write the output to a new VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(outvtk)
    writer.Write()

#convert ply to vtk
    #for NeRF output
def convert_ply_to_vtk(input, output):
    # Load the PLY file using trimesh
    mesh = trimesh.load_mesh(str(input))

    # Extract the vertices and faces (and normals) from the mesh
    vertices = mesh.vertices
    faces = mesh.faces.reshape(-1, 3)
    normals = mesh.vertex_normals

    # Create a VTK PolyData object
    polydata = vtk.vtkPolyData()

    # Create a vtkPoints object and set the vertex coordinates
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(vertices))
    polydata.SetPoints(vtk_points)

    # Create a vtkCellArray to store the faces
    vtk_faces = vtk.vtkCellArray()

    # Add the faces to the vtkCellArray
    for face in faces:
        vtk_faces.InsertNextCell(3, [int(face[0]), int(face[1]), int(face[2])])

    polydata.SetPolys(vtk_faces)

    #Add the normals to the VTK data as well
    vtk_normals = vtk.vtkFloatArray()
    vtk_normals.SetName("Normals")
    vtk_normals.SetNumberOfComponents(3)

    for normal in normals:
        vtk_normals.InsertNextTuple3(normal[0], normal[1], normal[2])

    polydata.GetPointData().SetNormals(vtk_normals)

    # Add color to the vertices
    colors = mesh.visual.vertex_colors  # Assuming color data exists in the PLY file

    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetName("Colors")
    vtk_colors.SetNumberOfComponents(3)

    for color in colors:
        vtk_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))

    polydata.GetPointData().SetScalars(vtk_colors)

    # Write the PolyData to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(polydata)
    writer.Write()


def calc_fiducial_distances_hip_H1(rootdir, obj_name="hip", static_name="CT_centered_fiducials.vtk"):

    def sort_by_proximity(v1, v2):
        #match the fiducials
        v2_match_list = []
        dists = []
        for i in range(v1.shape[0]):
            mindist = 1000000
            match = (0,0,0)
            
            for j in range(v2.shape[0]):
                dist = np.linalg.norm(v1[i]-v2[j])
                if dist < mindist:
                    match = v2[j]
                    mindist = dist
            v2_match_list.append(match)
            dists.append(mindist)
        return np.stack(v2_match_list), dists

    static = rootdir/("{}".format(static_name))
    _, static_fiducials, _, _ = registration.get_points_vtk(static)

    all_dists = []

    for i in range(1, 9, 1):
        if obj_name == "hip":
            surf_fiducials_file = rootdir/("{}{}".format(obj_name, i))/("registered_fiducials.vtk")
            if not surf_fiducials_file.exists():
                surf_fiducials_file = rootdir/("{}{}_fiducials_scaled_registered.vtk".format(obj_name, i))
        else:
            surf_fiducials_file = rootdir/("{}{}_centered_fiducials_registered.vtk".format(obj_name, i))
            #print(surf_fiducials_file)
        if not surf_fiducials_file.exists():
            continue
        _, surf_fiducials, _, _ = registration.get_points_vtk(surf_fiducials_file)
        #print(surf_fiducials)
        matched_fiducials, dists = sort_by_proximity(np.stack(static_fiducials), np.stack(surf_fiducials))

        all_dists.append((i, dists))
    return all_dists

def get_component_with_point(input, output, coords=None, point_id=None):

    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()

    #get the coordinates of the point with the point ID
    if coords:
        point_coords = coords
    elif point_id:
        point_coords = reader.GetOutput().GetPoint(point_id)
    else:
        print("Error: Need either coordinates or point ID as input")
        return None

    #Extr
    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(reader.GetOutput())
    connectivity_filter.SetExtractionModeToClosestPointRegion()
    connectivity_filter.SetClosestPoint(point_coords)
    connectivity_filter.Update()

    comp = connectivity_filter.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(comp)
    writer.Write()


#approximation for the scaling
    #fisrt, find the point that is furthest from the centroid
    #then, find the point that is furthest from this point
def appx_max_distance_mesh(points):

    def calc_centroid_vtk(points):
        x, y, z = 0, 0, 0
        num_points = points.GetNumberOfPoints()
        for i in range(num_points):
            point = points.GetPoint(i)
            x += point[0]
            y += point[1]
            z += point[2]
        x /= num_points
        y /= num_points
        z /= num_points
        return (x, y, z)

    centroid = calc_centroid_vtk(points)
    print(centroid)
    #centroid = np.mean(points, axis=0)
    num_points = points.GetNumberOfPoints()
    longest1 = 0
    point1 = None
    
    #First find the point that is furthest from the centroid
    for i in range(num_points):
        point = point1 = points.GetPoint(i)
        dist = vtk.vtkMath.Distance2BetweenPoints(point, centroid)
        if dist > longest1:
            longest1 = dist
            point1 = point
    
    #now, find the point that is furthest from this point
    longest2 = 0
    point2 = None
    for i in range(num_points):
        point = points.GetPoint(i)
        dist = vtk.vtkMath.Distance2BetweenPoints(point1, point)
        if dist > longest2:
            longest2 = dist
            point2 = point
    
    return (point1, point2), np.sqrt(longest2)


def brute_force_get_max_dist(input, step):

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    num_pts = points.GetNumberOfPoints()

    maxdist = 0
    pair = None
    for i in tqdm(range(0, num_pts, step)):
        point1 = points.GetPoint(i)
        for j in range(i+1, num_pts, step):
            point2 = points.GetPoint(j)
            dist =  vtk.vtkMath.Distance2BetweenPoints(point1, point2)
            if dist > maxdist:
                pair = (point1, point2)
                maxdist = dist
    return pair, maxdist


#read in vtk file and return the polydata
def get_polydata_vtk(input):
    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(input))
    reader.Update()
    return reader.GetOutput()

#read in a csv to get the point IDs of fiducials, and then gets the corresponding points on the ICP surface
    #outputs these fiducials as a pointcloud
def get_ICP_moved_fiducials(csv, vtkin, vtkout):
    #given point IDs in a csv file, creates a vtk file of the corresponding points from an ICP registered surface
    def get_pointIDs(csv):
        df = pd.read_csv(csv)
        ids = []
        for row in df.iterrows():
            id = row[1]['Point ID']
            ids.append(id)
        return ids
    
    ids = get_pointIDs(csv)
    #now, read in the vtk points for the ICP registered vtk
    polydata = get_polydata_vtk(vtkin)
    points = polydata.GetPoints()
    fiducials = []
    for id in ids:
        point = points.GetPoint(int(id))
        fiducials.append(np.array(point))
    #print(fiducials)
    registration.create_vtk_pointcloud(fiducials, vtkout)

###
# Related to fiducials
###

#given the CT with the markers and the segmentation from Peter's code, places the markers onto the segmentation
#CT has markers that do not get IDed in the segmentation from Peter's code
    #this will add them to the segmentation
def place_markers_on_seg(CT, SEG, OUT):
    ct_f = nib.load(CT)
    ct = ct_f.get_fdata()

    seg_f = nib.load(SEG)
    seg = seg_f.get_fdata()

    #get the locations where the markers are
    markers_loc = np.where(ct > 3000)

    #create a new segmentation
    new_seg = seg.copy()
    new_seg[markers_loc] = 1

    #now output the new nifti segmentation
    new_seg_f = nib.Nifti1Image(new_seg, seg_f.affine, header=seg_f.header)

    # Save the modified NIfTI data back to a new file
    nib.save(new_seg_f, OUT)

#move the mesh with markers on it into the centroid of the mesh without markers
    #will allow us to identify fiducials on the mesh without markers
def align_markered_to_centroid(vtk_seg, vtk_mark, vtk_seg_moved, vtk_mark_moved):
    # Load the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_seg))
    reader.Update()

    reader2 = vtk.vtkPolyDataReader()
    reader2.SetFileName(str(vtk_mark))
    reader2.Update()

    # Get the output polydata
    polydata = reader.GetOutput()
    polydata2 = reader2.GetOutput()

    # Get the points (vertices) from the polydata
    points = polydata.GetPoints()
    points2 = polydata2.GetPoints()

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

    #and from the second mesh
    num_points2 = points2.GetNumberOfPoints()
    for i in range(num_points2):
        point = points2.GetPoint(i)
        aligned_point = [p - c for p, c in zip(point, centroid)]
        points2.SetPoint(i, aligned_point)

    # Save the aligned points to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(vtk_seg_moved))
    writer.SetInputData(polydata)
    writer.Write()

    writer2 = vtk.vtkPolyDataWriter()
    writer2.SetFileName(str(vtk_mark_moved))
    writer2.SetInputData(polydata2)
    writer2.Write()


#these two convert the paraview output csv points into a vtk file of points
def create_vtk_file_from_points(points, output):
    #given a set of 3 fiducials, creates a VTK file of the 3 points
    #print([x for x in [coord for coord in points]])

    points = [np.array(point) for point in points]
    points = np.stack(points).flatten()
    print(points)

    with open(output, "w") as vtk_file:
        vtk_file.write("# vtk DataFile Version 5.1\n")
        vtk_file.write("vtk output\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")
        vtk_file.write("POINTS 3 float\n")
        vtk_file.write(" ".join(str(coord) for coord in points.tolist()) + "\n")
        vtk_file.write("VERTICES 4 3\n")
        vtk_file.write("OFFSETS vtktypeint64\n")
        vtk_file.write("0 1 2 3\n")
        vtk_file.write("CONNECTIVITY vtktypeint64\n")
        vtk_file.write("0 1 2\n")

def create_vtk_file_from_csv(csv, output):
    points = registration.get_fiducials_from_csv(csv)
    create_vtk_file_from_points(points, output)
##

#given a registered surface and a refgerence surface (VTK files), outputs a new VTK file with the surface distance array added to each point
    #mostly for visualization purposes
def create_vtk_with_surf_distances(testin, testref, testout):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(testin)
    reader.Update()

    polydata = reader.GetOutput()

    points=polydata.GetPoints()
    num_pts = points.GetNumberOfPoints()

    #new data array for surface distance
    dist_array = vtk.vtkDoubleArray()
    #dist_array.SetNumberOfComponents(3)
    dist_array.SetNumberOfComponents(1)
    dist_array.SetName("SurfDist")

    #get the distances
    (_, distances, _) = registration.compute_closest_points_on_surface(testin, testref)

    #loop through all the points and color by the distance from the static surface
        #TEST: color by coordinate
    for i in range(num_pts):
        point = points.GetPoint(i)
        data = distances[i]
        #dist_array.InsertNextTuple3(data[0], data[1], data[2])
        dist_array.InsertNextValue(data)

    #now assign the array to the polydata
    polydata.GetPointData().AddArray(dist_array)

    #now save to a new vtk file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(testout)
    writer.SetInputData(polydata)
    writer.Write()



#given a polydata object and a np array of points (that is the same length), replaces the points in polydata with the new array of points
def replace_points(polydata, points):
    
    vtk_new_points = vtk.vtkPoints()
    vtk_new_points.SetData(vtk.util.numpy_support.numpy_to_vtk(points))

    polydata.SetPoints(vtk_new_points)

    return polydata


#given a csv file of points exported from paraview and the vtk file from which they were selected, exports a vtk file
#with the normals, colors, vertices, and cell connectivity for all of the selected points
    #alternatively, just look at the pointIDs
def segment_vtk_from_csv(csv, invtk, outvtk):
    #read in csv
    csv_pts = pd.read_csv(csv)

    #parse the csv to get point data (coordinates, colors, normals)
    cols_to_keep = 'Point ID,Colors_0,Colors_1,Colors_2,Normals_0,Normals_1,Normals_2,Points_0,Points_1,Points_2'.split(',')
    pointIDs_to_keep = csv_pts['Point ID'].values

    #from the selected points, keep the faces and vertices
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(invtk)
    reader.Update()
    polydata = reader.GetOutput()

    #get the points and faces and colors and normals
    points = polydata.GetPoints()
    faces = polydata.GetPolys()
    point_data = polydata.GetPointData()
    colors = point_data.GetArray("Colors")
    normals = point_data.GetArray("Normals")

    #iterate through the faces. If all of the points are included in the pointIDs to keep, then add that face, the vertices,
    # and the vertex information (colors/normals) to the new polydata
    filtered_points = vtk.vtkPoints()
    filtered_faces = vtk.vtkCellArray()

    filtered_colors = vtk.vtkUnsignedCharArray()
    filtered_colors.SetName("Colors")
    filtered_colors.SetNumberOfComponents(3)

    vtk_normals = vtk.vtkFloatArray()
    vtk_normals.SetName("Normals")
    vtk_normals.SetNumberOfComponents(3)

    faces = polydata.GetPolys()
    faces.InitTraversal()

    old_to_new = {} #dictionaary from old indexes to new indexes
    
    #iterate over the points first
    for i,id in enumerate(pointIDs_to_keep):
        #get the information for that point
        point = points.GetPoint(id)
        color_tuple = colors.GetTuple(id)
        normal_tuple = normals.GetTuple(id)

        #add the mapping from new to old
        old_to_new[id] = i

        #add the point and information to the respective arrays
        filtered_points.InsertNextPoint(point)
        filtered_colors.InsertNextTuple3(color_tuple[0], color_tuple[1], color_tuple[2])
        vtk_normals.InsertNextTuple3(normal_tuple[0], normal_tuple[1], normal_tuple[2])
    
    #now iterate over the faces
    while True:
        face = vtk.vtkIdList()
        if faces.GetNextCell(face) == 0:
            break
        # Check if all face vertices are in the vertex map
        #if all(vertex_map.get(face.GetId(j)) is not None for j in range(3)):
        if all(old_to_new.get(face.GetId(j)) is not None for j in range(3)):
            filtered_face = vtk.vtkIdList()
            #all the points are included, so get all the point information and add it to the respective arrays
            for j in range(3):
                vertex_id = face.GetId(j)
                filtered_face.InsertNextId(old_to_new[vertex_id])
            filtered_faces.InsertNextCell(filtered_face)

    #now set all the faces and vertices to a new polydata object
    filtered_polydata = vtk.vtkPolyData()
    filtered_polydata.SetPoints(filtered_points)
    filtered_polydata.GetPointData().SetNormals(vtk_normals)
    filtered_polydata.GetPointData().SetScalars(filtered_colors)
    filtered_polydata.SetPolys(filtered_faces)

    #now output the new vtk file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(outvtk)
    writer.SetInputData(filtered_polydata)
    writer.Write()


#given a vtk file, output the convex hull of the pointcloud
def make_convex_hull(input_surf, output):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_surf)
    reader.Update()

    polydata = reader.GetOutput()
    point_list = polydata.GetPoints()
    points = np.array([point_list.GetPoint(i) for i in range(point_list.GetNumberOfPoints())])

    #calculate the convex hull
    convex_hull = ConvexHull(points)
    hull_pt_idxs = convex_hull.vertices
    hull_points = points[hull_pt_idxs]
    faces = convex_hull.simplices
    #the indices of faces are for the original point indexes. Need to convert the old indices to the new ones
    old_to_new = {}
    for i,idx in enumerate(hull_pt_idxs):
        old_to_new[idx] = i

    hull_faces = []
    for face in faces:
        new_face = []
        for pt in face:
            new_face.append(old_to_new[pt])
        hull_faces.append(np.array(new_face))
    hull_faces = np.stack(hull_faces)

    #create the vtk object
        #points
    vtkPoints = vtk.vtkPoints()
    for point in hull_points:
        vtkPoints.InsertNextPoint(point)

        #faces
    vtkCells = vtk.vtkCellArray()
    for simplex in hull_faces:
        vtkCells.InsertNextCell(len(simplex), simplex) #saying that we are inserting a triangle with the connectivity in "simplex"


    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtkPoints)
    polydata.SetPolys(vtkCells)

    #write it to a file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output)
    writer.SetInputData(polydata)
    writer.Write()
















    #need to do the faces and the vertices separately


        ####AHHHHHH figure it out tomorrow. Need to make sure that the point indices are being inserted correctly
            #probably do the points first and then the faces
            #for the points, create a mapping (like in the color segmentation) of old to new point IDs
            #then, for each face in the oldd polydata, if all the points are ones to keep:
                #use the mapping to insert a new face whose pointIDs are the new ones that we get from the mapping


# def calculate_mesh_weighted_surface_distance()

#functions for calculating the maximum distance between two points on a mesh
    #uses the convex hull method and Graham Scan
# def rotating_calipers(points):

#     def cross_product(a, b, c):
#         #given 3 points, calculate the cross product between the 2 vectors they form
#         v1 = b - a
#         v2 = c - b
#         return 


#     def convex_hull(points):
#         #assumes that the points are in a list?
#         points.sort(key=lambda p: (p[0], p[1], p[2]))
#         hull = []
        
#         for i in range(len(points)):

#             while len(hull) >= 2 and cross_product(hull[-2], hull[-1], points[i])