import open3d as o3d


def test_upsampling(mesh: o3d.geometry.TriangleMesh):
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    print(len(pcd.points))

    o3d.visualization.draw(pcd)
    print(pcd)

    upsampled_pcd = mesh.sample_points_poisson_disk(number_of_points=50000)
    print(upsampled_pcd)

    o3d.visualization.draw(upsampled_pcd)


def main():
    mesh = o3d.io.read_triangle_mesh("./samples/target.stl")
    test_upsampling(mesh)


main()
