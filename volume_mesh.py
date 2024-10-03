import meshio
import numpy as np
import vtk
import pdb  # Optional: If you're using pdb for debugging
import os

def read_vtk(filename):
    """Read a VTP file and extract the surface mesh."""
    if not os.path.exists(filename):
        raise ValueError("File does not exist: " + filename)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    cells = polydata.GetPolys()

    if points is None:
        raise ValueError("Error reading file: " + filename)

    vertices = []
    faces = []

    # Extract points
    for i in range(points.GetNumberOfPoints()):
        vertices.append(points.GetPoint(i))

    pdb.set_trace()
    # Extract faces
    for i in range(cells.GetNumberOfCells()):
        cell = cells.GetCell(i)
        face = []
        for j in range(cell.GetNumberOfPoints()):
            face.append(cell.GetPointId(j))
        faces.append(face)

    return np.array(vertices), np.array(faces)

def generate_volume_mesh(vertices, faces, volume_size):
    """Generate a volume mesh from the given surface mesh."""
    geom = pygmsh.built_in.Geometry()

    # Add the surface mesh
    surface = geom.add_polygon(vertices, mesh_size=volume_size)

    # Generate a volume mesh
    volume = geom.add_box([0, 0, 0], [1, 1, 1])  # Modify as needed

    mesh = pygmsh.generate_mesh(geom)

    return mesh

def export_vtu(mesh, output_file):
    """Export the mesh to VTU format."""
    meshio.write(output_file, mesh)

def main(vtp_file, output_file, volume_size=0.1):
    # Step 1: Read the VTP surface mesh
    vertices, faces = read_vtk(vtp_file)

    # Step 2: Generate the volume mesh
    volume_mesh = generate_volume_mesh(vertices, faces, volume_size)

    # Step 3: Export the volume mesh to VTU format
    export_vtu(volume_mesh, output_file)

# Call the main function (you can pass appropriate file paths as needed)
if __name__ == "__main__":
    vtp_file = "/home/nrosete/repos/svFSIplus/tests/cases/struct/LV_Holzapfel_passive/mesh/mesh-complete.exterior.vtp"  # Replace with actual .vtp file path
    output_file = "output_mesh.vtu"  # Replace with desired output .vtu file path
    main(vtp_file, output_file)
