from skimage.measure import marching_cubes
import nibabel as nib
from skimage.morphology import binary_closing, ball
import numpy as np
import sys
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tetgen
from scipy.ndimage import binary_fill_holes

def extract_meshes_from_segm(segm, mesh_filename, affine):
    print(f"Unique values in segmentation before processing: {np.unique(segm)}")
    segm = np.round(segm).astype(np.uint8)
    label_1 = np.zeros_like(segm, dtype=np.uint8)
    label_1[segm > 0] = 1
    unique_vals = np.unique(label_1)
    print(f"Unique values in label_1: {unique_vals}")
    if len(unique_vals) == 1 and unique_vals[0] == 0:
        raise ValueError("Segmentation contains only zeros. Check input NIfTI file and labels.")
    
    footprint = ball(7)
    label_1 = binary_closing(label_1, footprint)
    label_1 = binary_fill_holes(label_1)
    label_1 = extract_largest_component(label_1)
    
    plt.imshow(label_1[:, :, label_1.shape[2] // 2], cmap="gray")
    plt.title("Middle Slice of Processed Segmentation")
    plt.show()
    
    verts, faces, _, _ = marching_cubes(label_1, level=0.5)
    polydata = pv.PolyData(verts, np.hstack((np.full((faces.shape[0], 1), 3), faces)))
    polydata = polydata.triangulate()
    polydata = polydata.extract_surface()
    cleaned_polydata = polydata.clean(tolerance=1e-6)
    
    if not cleaned_polydata.is_manifold:
        print("Warning: Surface is not manifold. Attempting further repair.")
        cleaned_polydata = cleaned_polydata.extract_largest()  # Keep only the largest component
        cleaned_polydata = cleaned_polydata.clean(tolerance=1e-6)
    
    if not cleaned_polydata.is_manifold:
        raise RuntimeError("Failed to repair surface. Ensure the input segmentation is valid.")
    
    tg = tetgen.TetGen(cleaned_polydata)
    tg.tetrahedralize()
    
    tet_mesh = trimesh.Trimesh(vertices=tg.points, faces=tg.faces, process=False)
    tet_mesh.apply_transform(affine)
    print(f'Mesh is watertight: {tet_mesh.is_watertight}')
    save_mesh(tet_mesh, mesh_filename)
    visualize_mesh(tet_mesh.vertices, tet_mesh.faces)
    
    return tet_mesh

def extract_largest_component(segmentation):
    from skimage.measure import label, regionprops
    labeled_seg = label(segmentation)
    regions = regionprops(labeled_seg)
    if not regions:
        raise ValueError("No connected components found in segmentation.")
    largest_region = max(regions, key=lambda r: r.area)
    largest_component = labeled_seg == largest_region.label
    return largest_component.astype(np.uint8)

def save_mesh(mesh, filename):
    mesh.export(filename)

def load_img_file(file):
    data = nib.load(file)
    affine = data.affine
    return data.get_fdata(), affine

def visualize_mesh(vertices, faces):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(vertices[faces], alpha=1.0, facecolor='white', edgecolor='b', linewidth=0.03)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, vertices[:, 0].max())
    ax.set_ylim(0, vertices[:, 1].max())
    ax.set_zlim(0, vertices[:, 2].max())
    ax.axis('off')
    plt.title("Extracted Mesh Visualization")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 voxel2surface.py <input_nifti_file> <output_mesh_file>")
        sys.exit(1)
    input_nifti_file = sys.argv[1]
    output_mesh_file = sys.argv[2]
    segm, affine = load_img_file(input_nifti_file)
    extract_meshes_from_segm(segm, output_mesh_file, affine)

