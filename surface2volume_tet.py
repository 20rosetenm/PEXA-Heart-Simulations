import meshio
import numpy as np
import pyvista as pv
from tetgen import TetGen  # Proper import

# Step 1: Load the surface mesh using PyVista
surface_mesh = pv.read("mesh-complete.exterior.vtp")

# Step 2: Create a TetGen object from the surface mesh
tet = TetGen(surface_mesh)

# Step 3: Perform tetrahedralization with the desired switches
# Switches explanation:
# 'pq1.2' = piecewise linear complex, min quality ratio of 1.2
# '/20Ya2.0' = minimum dihedral angle of 20 degrees, no bisection, max volume constraint of 2.0
tet.tetrahedralize(switches='pq1.2/20Ya2.0')

# Step 4: Convert the tetrahedral mesh back into a PyVista grid
tetgen_pv = tet.grid

# Optional: Visualize the result in PyVista
tetgen_pv.plot()

# Step 5: Save the tetrahedralized mesh in a desired format
tetgen_pv.save("tetrahedral_volume_mesh.vtu")

