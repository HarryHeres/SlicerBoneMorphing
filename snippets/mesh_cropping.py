import numpy as np
import open3d as o3d
import os

MESH_SOURCE_DIR = ""
MESH_OUTPUT_DIR = ""


def crop(mesh_filename: str, gender: str):
    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_SOURCE_DIR, gender, mesh_filename))

    obb = mesh.get_oriented_bounding_box()

    extents = np.asarray(obb . extent)
    major_axis_index = np.argmax(extents)
    major_axis_extent = extents[major_axis_index]

    cropped_extent = extents.copy()
    cropped_extent[major_axis_index] /= 3

    cropped_center = obb.center + obb.R[:, major_axis_index] * (major_axis_extent / 3)

    cropped_obb = o3d.geometry.OrientedBoundingBox(cropped_center, obb .R, cropped_extent)
    proximal_mesh = mesh.crop(cropped_obb)
    proximal_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([proximal_mesh])
    o3d.io.write_triangle_mesh(mesh=proximal_mesh, filename=os.path.join(MESH_OUTPUT_DIR, gender, mesh_filename))
