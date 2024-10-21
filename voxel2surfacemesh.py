import argparse
import pyvista as pv
import numpy as np
import SimpleITK as sitk

# Step 1: Load Input File/Segmentations
def load_segmentation(input_file):
    """Load segmentation from .nrrd or .nii/.nii.gz file."""
    print(f"Loading segmentation from {input_file}...")
    img = sitk.ReadImage(input_file)  # Read the segmentation file
    seg = sitk.GetArrayFromImage(img)  # Convert to NumPy array (shape: [D, H, W])

    # Extract spacing and origin (for accurate rendering)
    spacing = img.GetSpacing()[::-1]  # Reverse for [Z, Y, X] order
    origin = img.GetOrigin()
    
    print(f"Segmentation loaded with shape {seg.shape}, spacing {spacing}, origin {origin}.")
    return seg, spacing, origin

# Step 2: Define/Paramaterize Segmentation Data
def seg_to_polydata(seg, isolevel=0.5, spacing=[1, 1, 1], origin=[0, 0, 0], 
                    smooth=True, volume_threshold=0.0, subdivide=0):
    """Convert segmentation to surface mesh using PyVista."""
    if seg.sum() == 0:  # Check if the segmentation is empty
        print("Warning: Segmentation data is empty. Returning empty PolyData.")
        return pv.PolyData()

# Step 3: Create PyVista ImageData from segmentation
    seg_pv = pv.ImageData(dimensions=seg.shape, spacing=spacing, origin=origin)
    seg_pv.point_data['values'] = seg.flatten(order='F')

# Step 4: Mesh Cleanup
    # Generate the surface mesh using contouring
    mesh_pv = seg_pv.contour([isolevel])

    # Apply volume filtering if required
    if volume_threshold > 0:
        bodies = mesh_pv.split_bodies()
        filtered_bodies = [body for body in bodies if body.extract_surface().volume > volume_threshold]
        if filtered_bodies:
            mesh_pv = pv.MultiBlock(filtered_bodies).combine()
        else:
            print("Warning: No bodies passed the volume threshold. Returning empty PolyData.")
            return pv.PolyData()

    # Optional: Smooth and subdivide the mesh
    if smooth:
        mesh_pv = mesh_pv.smooth(n_iter=5, relaxation_factor=0.01)
    if subdivide > 0:
        mesh_pv = mesh_pv.subdivide(subdivide)

    return mesh_pv

def main(input_file, output_file):
# Step 5: Load segmentation data from input - main
    seg, spacing, origin = load_segmentation(input_file)

# Step 6: Convert segmentation to surface mesh - main
    print("Converting segmentation to surface mesh...")
    mesh = seg_to_polydata(seg, spacing=spacing, origin=origin)

# Step 7: Save the surface mesh to the output file - main
    print(f"Saving surface mesh to {output_file}...")
    mesh.save(output_file)

# Step 8: Display the surface mesh
    print("Displaying surface mesh...")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', show_edges=True)
    plotter.show()

if __name__ == "__main__":
    # Setup argument parser to accept input and output files
    parser = argparse.ArgumentParser(description="Convert 3D Slicer segmentation to surface mesh.")
    parser.add_argument("input_file", type=str, help="Path to the input segmentation (.nrrd, .nii, .nii.gz)")
    parser.add_argument("output_file", type=str, help="Path to the output surface mesh (.vtp)")
    args = parser.parse_args()

# Step 9: Run the main function
    main(args.input_file, args.output_file)

#Run script in ubuntu: python3 voxel2surfacemesh.py [filename] 
