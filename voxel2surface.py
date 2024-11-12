import numpy as np
import trimesh
from skimage.measure import marching_cubes
import os
import nibabel as nib
from skimage.morphology import binary_closing, ball
import sys

def extract_meshes_from_segm(segm, mesh_filename):
    # Myocardium label extraction (assuming label 1 for segmentation)
    label_1 = np.zeros_like(segm)
    label_1[segm == 1] = 1

    # Morphological closing to handle potential holes in segmentation
    footprint = ball(3)
    label_1 = binary_closing(label_1, footprint)

    # Marching cubes to extract surface meshes
    verts, faces, normals, values = marching_cubes(label_1, 0, gradient_direction='ascent', allow_degenerate=False, step_size=2)

    # Use trimesh for saving the mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    print(f'Mesh is watertight: {mesh.is_watertight}')
    print(f'Number of vertices: {verts.shape[0]}')

    # Save the mesh
    save_mesh(mesh, mesh_filename)
    return mesh

def save_mesh(mesh, filename):
    # Save the mesh in PLY format
    mesh.export(filename)

def load_img_file(file):
    # Load image using nibabel for NIfTI files
    data = nib.load(file)
    return data.get_fdata()

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python3 voxel2surface.py <input_nifti_file> <output_mesh_file>")
        sys.exit(1)

    input_nifti_file = sys.argv[1]
    output_mesh_file = sys.argv[2]

    # Load input NIfTI file
    segm = load_img_file(input_nifti_file)

    # Extract mesh and save it to the output file
    extract_meshes_from_segm(segm, output_mesh_file)

