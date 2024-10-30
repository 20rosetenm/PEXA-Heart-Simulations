import argparse
import pyvista as pv
import numpy as np
import SimpleITK as sitk

# Step 1: Load Input File/Segmentations
def load_segmentation(input_file):
    """Load segmentation from .nrrd or .nii/.nii.gz file."""
    print(f"Loading segmentation from {input_file}...")
    img = sitk.ReadImage(input_file)  # Read the segmentation file
    seg = sitk.GetArrayFromImage(img)  # Convert to NumPy array (shape: [T, Z, Y, X])

    # Extract spacing and origin
    spacing = img.GetSpacing()[::-1]  # Reverse to [Z, Y, X]
    origin = img.GetOrigin()

    print(f"Segmentation loaded with shape {seg.shape}, spacing {spacing}, origin {origin}.")
    return seg, spacing, origin

# Step 2: Convert Segmentation to PyVista PolyData
def seg_to_polydata(seg, isolevel=0.5, spacing=[1, 1, 1], origin=[0, 0, 0],
                    smooth=True, volume_threshold=0.0, subdivide=0):
    """Convert a single 3D segmentation volume to surface mesh."""
    # Extract a single 3D volume if segmentation is 4D
    if seg.ndim == 4:
        print("Extracting the first volume from the 4D segmentation...")
        seg = seg[0]  # Extract the first volume (adjust index if needed)

    # Check if segmentation data is empty
    if seg.sum() == 0:
        print("Warning: Segmentation data is empty. Returning empty PolyData.")
        return pv.PolyData()

    # Create PyVista ImageData (dimensions = [NX, NY, NZ])
    dims = seg.shape[::-1]  # Reverse to match PyVista's [NX, NY, NZ] format
    seg_pv = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)

    # Flatten the single 3D volume with 'F' order
    seg_pv.point_data['values'] = seg.flatten(order='F')

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

# Step 3: Main Function
def main(input_file, output_file):
    # Load segmentation data from input
    seg, spacing, origin = load_segmentation(input_file)

    # Convert segmentation to surface mesh
    print("Converting segmentation to surface mesh...")
    mesh = seg_to_polydata(seg, spacing=spacing, origin=origin)

    # Save the surface mesh to the output file
    print(f"Saving surface mesh to {output_file}...")
    mesh.save(output_file)

# Step 4: Command-Line Interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert voxel data to surface mesh.")
    parser.add_argument("input_file", help="Path to the input NRRD/NIfTI file.")
    parser.add_argument("output_file", help="Path to the output mesh file.")
    args = parser.parse_args()

    # Run the main function
    main(args.input_file, args.output_file)
