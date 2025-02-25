import numpy as np
import sys
import nrrd
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes, label, regionprops
from skimage.morphology import binary_closing, ball
from scipy.ndimage import binary_fill_holes
import tetgen

def extract_meshes_from_segm(segm, mesh_filename):
    print(f"Unique values in segmentation before processing: {np.unique(segm)}")
    
    segm = np.round(segm).astype(np.uint8)
    label_1 = (segm > 0).astype(np.uint8)
    
    unique_vals = np.unique(label_1)
    if len(unique_vals) == 1 and unique_vals[0] == 0:
        raise ValueError("Segmentation contains only zeros. Check input NRRD file.")
    
    label_1 = binary_closing(label_1, ball(5))
    label_1 = binary_fill_holes(label_1)
    label_1 = extract_largest_component(label_1)
    
    plt.imshow(label_1[:, :, label_1.shape[2] // 2], cmap="gray")
    plt.title("Middle Slice of Processed Segmentation")
    plt.show()
    
    verts, faces, _, _ = marching_cubes(label_1, level=0.5)
    polydata = pv.PolyData(verts, np.hstack((np.full((faces.shape[0], 1), 3), faces)))
    polydata = polydata.triangulate().extract_surface().clean(tolerance=1e-4).extract_largest()
    polydata = fill_holes_except_largest(polydata)
    
    if not polydata.is_manifold:
        print("Warning: Surface is not manifold. Attempting further repair.")
        polydata = polydata.clean(tolerance=1e-3).extract_largest()
    
    if not polydata.is_manifold:
        raise RuntimeError("Failed to repair surface. Ensure the input segmentation is valid.")
    
    polydata = polydata.triangulate()
    print(f"Number of faces before TetGen: {polydata.n_faces}")
    
    tg = tetgen.TetGen(polydata)
    try:
        tg.tetrahedralize()
    except RuntimeError:
        print("TetGen tetrahedralization failed. Check mesh integrity.")
        return None
    
    tet_vertices = tg.node if hasattr(tg, 'node') else None
    tet_faces = tg.facet if hasattr(tg, 'facet') else None
    if tet_vertices is None or tet_faces is None:
        raise RuntimeError("TetGen output is invalid. Ensure input surface is correct.")
    
    tet_mesh = trimesh.Trimesh(vertices=tet_vertices, faces=tet_faces, process=False)
    save_mesh(tet_mesh, mesh_filename)
    visualize_mesh(tet_mesh.vertices, tet_mesh.faces)
    
    return tet_mesh

def extract_largest_component(segmentation):
    labeled_seg = label(segmentation)
    regions = regionprops(labeled_seg)
    if not regions:
        raise ValueError("No connected components found in segmentation.")
    
    largest_region = max(regions, key=lambda r: r.area)
    return (labeled_seg == largest_region.label).astype(np.uint8)

def fill_holes_except_largest(mesh):
    connectivity = mesh.connectivity()
    connectivity.set_active_scalars("RegionId")
    hole_sizes = connectivity.point_data['RegionId']
    largest_hole_id = np.bincount(hole_sizes).argmax()
    filled_mesh = connectivity.threshold(largest_hole_id, invert=True)
    
    if not isinstance(filled_mesh, pv.PolyData):
        filled_mesh = filled_mesh.extract_surface()
    
    return filled_mesh.triangulate()

def save_mesh(mesh, filename):
    mesh.export(filename)

def load_img_file(file):
    data, _ = nrrd.read(file)  # Load .nrrd segmentation
    return data, np.eye(4)  # NRRD does not store affine, so return identity matrix

def visualize_mesh(vertices, faces):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    ax.add_collection3d(plt.Poly3DCollection(mesh.triangles, alpha=1.0, facecolor='white', edgecolor='b', linewidth=0.03))
    ax.set_xlim(0, vertices[:, 0].max())
    ax.set_ylim(0, vertices[:, 1].max())
    ax.set_zlim(0, vertices[:, 2].max())
    ax.axis('off')
    plt.title("Extracted Mesh Visualization")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 voxel2surface.py <input_nrrd_file> <output_mesh_file>")
        sys.exit(1)

    input_nrrd_file = sys.argv[1]
    output_mesh_file = sys.argv[2]
    
    segm, _ = load_img_file(input_nrrd_file)
    extract_meshes_from_segm(segm, output_mesh_file)

