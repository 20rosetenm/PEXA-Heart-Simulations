import meshio
import numpy as np
import pyvista as pv

#Step 1: Define/Parametrize Segmentation Data
def seg_to_polydata(seg, isolevel=0.5, dims=None, spacing=[1,1,1], origin=[0,0,0], smooth=True, volume_threshold=0.0, subdivide=0, contour_method='contour'):
   # seg: np.ndarray of shape (H,W,D)
   if seg.sum() == 0: # check seg
     return pv.PolyData()

#Check Dimensions of Image Segmentations
   if dims is None:
     dims = seg.shape
   seg_pv = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
   seg_pv.point_data['values'] = seg.flatten(order='F')
   mesh_pv = seg_pv.contour([isolevel], method=contour_method)
   if volume_threshold != 0:
     bodies_orig = mesh_pv.split_bodies()
     multi_bodies = pv.MultiBlock([body for body in bodies_orig if body.extract_surface().volume>volume_threshold]) # filter ca2 chunks by volume
     if len(multi_bodies) > 0: # check after volume threshold
       mesh_pv = mesh_to_PolyData(*get_verts_faces_from_pyvista(multi_bodies.combine()))
     else:
       return pv.PolyData()

#Step 3: Create & Return Surface Mesh
   mesh_pv.clear_data()
   if smooth:
     mesh_pv = mesh_pv.smooth(n_iter=5, relaxation_factor=0.01) # smooth ca2 chunks
   if subdivide > 0:
     mesh_pv = mesh_pv.subdivide(subdivide) # makes ca2_smooth higher resolution, much better when removing nodes

   return mesh_pv
