#Edit script to take .ply instead of .vtp
import argparse
import pyvista as pv
import tetgen

def load_ply_file(input_file):
    """Load PLY file using PyVista."""
    print(f"Loading PLY file: {input_file}...")
    surface_mesh = pv.read(input_file)
    return surface_mesh

def check_mesh_properties(mesh):
    """Check and print mesh properties."""
    print(f"Mesh is manifold: {mesh.is_manifold}")
    print(f"Mesh is watertight: {mesh.is_all_triangles() and mesh.is_manifold}")
    print(f"Mesh has {mesh.n_cells} cells and {mesh.n_points} points.")

def repair_mesh(mesh):
    """Repair the mesh to make it more suitable for tetrahedralization."""
    print("Repairing the mesh...")
    # Clean the mesh (removes duplicate points, etc.)
    mesh = mesh.clean()

    # Fill small holes if they exist
    mesh = mesh.fill_holes(size=100.0)  # Adjust size based on expected hole size

    # Extract largest connected surface if there are multiple disconnected components
    mesh = mesh.extract_largest_surface()

    return mesh

def convert_to_volume_mesh(surface_mesh):
    """Convert a surface mesh to a volume mesh using TetGen."""
    print("Converting surface mesh to volume mesh...")

    # Create a TetGen instance
    tet = tetgen.TetGen(surface_mesh)

    # Perform tetrahedralization with some parameters
    try:
        nodes, elems = tet.tetrahedralize(order=1)  # Tetrahedralize with specified order
    except RuntimeError as e:
        print(f"Error during tetrahedralization: {e}")
        raise

    # Create a PyVista UnstructuredGrid
    volume_mesh = tet.grid

    return volume_mesh

def main(input_ply_file, output_vtu_file):
    # Load surface mesh from PLY file
    surface_mesh = load_ply_file(input_ply_file)

    # Check and repair the mesh
    check_mesh_properties(surface_mesh)
    surface_mesh = repair_mesh(surface_mesh)

    # Convert surface mesh to volume mesh
    try:
        volume_mesh = convert_to_volume_mesh(surface_mesh)
    except RuntimeError:
        print("Tetrahedralization failed even after repair. Please inspect the mesh.")
        return

    # Save the volume mesh to a VTU file
    print(f"Saving volume mesh to {output_vtu_file}...")
    volume_mesh.save(output_vtu_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert surface mesh to volume mesh.")
    parser.add_argument("input_ply_file", help="Path to the input PLY file.")
    parser.add_argument("output_vtu_file", help="Path to the output VTU file.")
    args = parser.parse_args()

    # Run the main function
    main(args.input_ply_file, args.output_vtu_file)

