import operator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import vtk
from tqdm import tqdm
import trimesh

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
    #tailered to green
def color_threshold_mesh(input, output, abs_threshold=[200,200,200], ratio_threshold=0.75, g_abs_thresh=90):

    def pass_abs_threshold(color, R, G, B):
        return color[0] < R and color[1] > G and color[2] < B
    
    def pass_ratio_threshold(color):
        red, green, blue = color[0], color[1], color[2]
        if green == 0:
            return False
        return red/green < ratio_threshold and blue/green < ratio_threshold and green > g_abs_thresh

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

    # Iterate over the points and colors
    R = abs_threshold[0]
    G = abs_threshold[1]
    B = abs_threshold[2]
    print("Creating mask based on color threshold of [{},{},{}]...".format(R, G, B))
    for i in range(polydata.GetNumberOfPoints()):
        color = color_array.GetTuple(i)

        if pass_abs_threshold(color, R, G, B) or pass_ratio_threshold(color):
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

#convert ply to vtk
    #for NeRF output
def convert_ply_to_vtk(input, output):
    # Load the PLY file using trimesh
    mesh = trimesh.load_mesh(str(input))

    # Extract the vertices and faces from the mesh
    vertices = mesh.vertices
    faces = mesh.faces.reshape(-1, 3)

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