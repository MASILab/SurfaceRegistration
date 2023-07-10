import operator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import vtk
from tqdm import tqdm

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
    reader.SetFileName(stl_file_path)
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
    writer.SetFileName(output)
    writer.SetInputData(polydata)
    writer.Write()
