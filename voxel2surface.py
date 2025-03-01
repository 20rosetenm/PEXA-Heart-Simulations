import numpy as np
import pyvista as pv
import tetgen
import trimesh
import argparse
import SimpleITK as sitk
import nibabel as nib
from skimage.measure import marching_cubes

def load_segmentation(file_path):
    """Loads a segmentation file (NIfTI or NRRD) and returns the array and affine matrix."""
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        img = nib.load(file_path)
        segm = img.get_fdata().astype(np.uint8)
        affine = img.affine  # Get affine transformation matrix
    else:  # NRRD format
        img = sitk.ReadImage(file_path)
        segm = sitk.GetArrayFromImage(img).astype(np.uint8)
        spacing = np.array(img.GetSpacing())[::-1]
        direction = np.array(img.GetDirection()).reshape(3, 3)
        origin = np.array(img.GetOrigin())

        # Construct affine matrix for NRRD
        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
    
    print(f"Loaded segmentation: {file_path}")
    print(f"Unique values in segmentation: {np.unique(segm)}")
    print(f"Affine matrix:\n{affine}")

    return segm, affine

def extract_meshes_from_segm(segm, output_mesh_file, affine, label_value=1):
    """Extracts surface mesh from a segmentation volume, applies affine transform, and saves it."""
    print(f"Processing label: {label_value}")

    # Convert segmentation to binary mask
    binary_mask = (segm == label_value).astype(np.uint8)
    print(f"Unique values in binary mask: {np.unique(binary_mask)}")

    # Run Marching Cubes to extract surface
    verts, faces, _, _ = marching_cubes(binary_mask, level=0.5)
    print(f"Marching Cubes Output - Vertices: {len(verts)}, Faces: {len(faces)}")

    # Convert to PyVista format
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])  # Add 3 for triangle format
    faces = faces.astype(np.int64).flatten()
    polydata = pv.PolyData(verts, faces)

    print(f"Initial mesh: {polydata.n_faces} faces")

    # Convert PyVista mesh to Trimesh format
    trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces.reshape(-1, 4)[:, 1:])

    # Apply affine transformation using trimesh
    trimesh_mesh.apply_transform(affine)
    print("Applied affine transformation to mesh.")

    # Convert back to PyVista
    transformed_polydata = pv.PolyData(trimesh_mesh.vertices, np.hstack([np.full((trimesh_mesh.faces.shape[0], 1), 3), trimesh_mesh.faces]).flatten())

    # Debugging: Save raw output before cleaning
    transformed_polydata.save("debug_mesh_before_cleaning.vtp")

    # Clean the mesh
    cleaned_polydata = transformed_polydata.clean(tolerance=1e-5, absolute=False)
    print(f"Mesh after cleaning: {cleaned_polydata.n_faces} faces")

    if cleaned_polydata.n_faces == 0:
        raise RuntimeError("Extracted surface has no faces. Check segmentation input.")

    # Check if the mesh is triangulated
    if not cleaned_polydata.is_all_triangles:
        print("Triangulating mesh...")
        cleaned_polydata.triangulate()

    print(f"Mesh is now triangulated: {cleaned_polydata.is_all_triangles}")

    # Save output mesh
    cleaned_polydata.save(output_mesh_file)
    print(f"Mesh saved to {output_mesh_file}")

    # Convert to TetGen format
    tg = tetgen.TetGen(cleaned_polydata)
    print("TetGen conversion successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a segmentation to a surface mesh.")
    parser.add_argument("input_segmentation", type=str, help="Path to the input segmentation file (.nii or .nrrd)")
    parser.add_argument("output_mesh_file", type=str, help="Path to save the output surface mesh (.vtp)")
    parser.add_argument("--label", type=int, default=1, help="Label value to extract (default: 1)")

    args = parser.parse_args()

    segm, affine = load_segmentation(args.input_segmentation)
    extract_meshes_from_segm(segm, args.output_mesh_file, affine, label_value=args.label)

