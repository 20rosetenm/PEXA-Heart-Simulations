import pyvista as pv
import trimesh
import numpy as np
import sys
import tetgen

def preprocess_with_trimesh(input_file):
    """Load and clean the mesh with trimesh."""
    print(f"Loading mesh: {input_file}")

    # Load the mesh using trimesh
    trimesh_mesh = trimesh.load(input_file)

    print("Fixing mesh winding...")
    trimesh.repair.fix_winding(trimesh_mesh)  # Fix inconsistent face winding

    print("Filling holes...")
    trimesh.repair.fill_holes(trimesh_mesh)   # Fill holes in the surface mesh

    print("Removing degenerate faces using new API...")
    trimesh_mesh.update_faces(trimesh_mesh.nondegenerate_faces())

    print("Fixing normals and ensuring consistent connectivity...")
    trimesh.repair.fix_normals(trimesh_mesh)  # Fix normal consistency

    print("Checking for non-manifold geometry...")
    
    # Identify non-manifold edges and vertices
    edge_count = {}
    for edge in trimesh_mesh.edges_unique:
        edge_tuple = tuple(sorted(edge))
        edge_count[edge_tuple] = edge_count.get(edge_tuple, 0) + 1

    non_manifold_edges = [edge for edge, count in edge_count.items() if count > 2]
    has_non_manifold_edges = len(non_manifold_edges) > 0
    has_non_manifold_vertices = not trimesh_mesh.is_watertight

    if has_non_manifold_edges or has_non_manifold_vertices:
        print("Mesh contains non-manifold geometry. Attempting to fix...")

        # Split into connected components and keep the largest one
        components = trimesh_mesh.split(only_watertight=False)
        if components:
            trimesh_mesh = max(components, key=lambda c: c.area)
            print(f"Mesh has {len(components)} components. Keeping the largest one.")
            trimesh_mesh.export("step_largest_component.ply")
        else:
            print("Error: No manifold component found.")
            return None

    # Export repaired mesh for inspection
    repaired_file = "repaired_mesh.ply"
    trimesh_mesh.export(repaired_file)
    print(f"Repaired mesh saved: {repaired_file}")

    # Load into PyVista for further processing
    return pv.read(repaired_file)

def check_mesh_properties(mesh):
    """Check properties of the mesh."""
    is_manifold = mesh.is_manifold
    is_surface = mesh.n_cells > 0 and mesh.is_all_triangles
    edges = mesh.extract_all_edges()
    boundary_edges = edges.extract_feature_edges(boundary_edges=True)
    is_watertight = boundary_edges.n_cells == 0

    print(f"Mesh is manifold: {is_manifold}")
    print(f"Mesh is watertight: {is_watertight}")
    print(f"Mesh is all triangles: {mesh.is_all_triangles}")
    print(f"Mesh is a surface mesh: {is_surface}")
    print(f"Mesh has {mesh.n_cells} cells and {mesh.n_points} points")

    if not is_manifold:
        print("Warning: Mesh is not manifold, which may cause issues in tetrahedralization.")

def generate_volume_mesh(surface_mesh):
    """Generate a volume mesh from a surface mesh using TetGen."""
    print("Generating volume mesh...")
    
    if not surface_mesh.is_manifold:
        print("Error: The mesh is not manifold and may not be suitable for tetrahedralization.")
        return None

    # Convert PolyData to TetGen-compatible format
    tet = tetgen.TetGen(surface_mesh)
    
    # Perform tetrahedralization
    tet.tetrahedralize()

    # Extract the volume mesh as a PyVista UnstructuredGrid
    volume_mesh = tet.grid
    
    return volume_mesh

def add_scalars_to_volume_mesh(volume_mesh):
    """Add scalar data to the volume mesh for visualization in ParaView."""
    center = volume_mesh.center
    distances = np.linalg.norm(volume_mesh.points - center, axis=1)
    volume_mesh.point_data['DistanceFromCenter'] = distances
    return volume_mesh

def main(input_ply_file, output_vtk_file):
    # Preprocess the surface mesh with trimesh
    surface_mesh = preprocess_with_trimesh(input_ply_file)
    if surface_mesh is None:
        print("Error: Failed to process the surface mesh.")
        return

    # Check the properties of the surface mesh
    check_mesh_properties(surface_mesh)

    # Generate the volume mesh (tetrahedralization)
    volume_mesh = generate_volume_mesh(surface_mesh)

    if volume_mesh:
        # Add scalar data (e.g., DistanceFromCenter) to the volume mesh
        volume_mesh = add_scalars_to_volume_mesh(volume_mesh)

        # Save the generated volume mesh with added scalar data
        volume_mesh.save(output_vtk_file)
        print(f"Volume mesh saved to: {output_vtk_file}")
    else:
        print("Volume mesh generation failed.")

# File paths from command-line arguments
if __name__ == "__main__":
    input_ply_file = sys.argv[1]
    output_vtk_file = sys.argv[2]
    # Run the main function
    main(input_ply_file, output_vtk_file)

# File path
input = "C:/Users/Nadin/OneDrive/Documents/Computational_Biomechanics_Lab/Repos/compbiomechproject/volume_meshes/PEXA12/PEXA12.vtk"
output = "C:/Users/Nadin/OneDrive/Documents/Computational_Biomechanics_Lab/Repos/compbiomechproject/volume_meshes/PEXA12/PEXA12.vtu"

#Read and convert
mesh = pv.read(input)
mesh.save(output)

print(f"Mesh converted to .vtu successfully.")