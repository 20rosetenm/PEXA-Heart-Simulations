import argparse
import meshio
import numpy as np
import pyvista as pv
from tetgen import TetGen  # Proper import

def main(input_file):
    # Step 1: Load the surface mesh using PyVista
    surface_mesh = pv.read(input_file)

    # Step 2: Create a TetGen object from the surface mesh
    tet = TetGen(surface_mesh)

    # Step 3: Perform tetrahedralization with the desired switches
    # Switches explanation:
    # 'pq1.2' = piecewise linear complex, min quality ratio of 1.2
    # '/20Ya2.0' = minimum dihedral angle of 20 degrees, no bisection, max volume constraint of 2.0
    tet.tetrahedralize(switches='pq1.2/20Ya2.0')

    # Step 4: Convert the tetrahedral mesh back into a PyVista grid
    tetgen_pv = tet.grid

    # PyVista Visulization
    tetgen_pv.plot()

    # Step 5: Save the tetrahedralized mesh in a desired format
    output_file = input_file.replace(".vtp", ".vtu")
    tetgen_pv.save(output_file)
    print(f"Saved tetrahedralized mesh to {output_file}")

if __name__ == "__main__":
    # Setup argument parser to accept input file from command line
    parser = argparse.ArgumentParser(description="Tetrahedralize a surface mesh.")
    parser.add_argument("input_file", type=str, help="Path to the input surface mesh file (e.g., .vtp)")
    args = parser.parse_args()

    # Call the main function with the provided input file
    main(args.input_file)

    #Call script in ubuntu: python3 surface2volume_tet.py [filename.vtp]
