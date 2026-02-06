import sys
import laspy
import numpy as np
import open3d as o3d


def load_lidar(file_path):
    """Load LAS file and print basic metadata."""
    print(f"\n--- Loading: {file_path} ---")

    with laspy.open(file_path) as fh:
        print(f"Points in file : {fh.header.point_count}")
        print(f"Point format  : {fh.header.point_format}")

    return laspy.read(file_path)


def offset_coordinates(las):
    """Center point cloud near origin for GPU-friendly processing."""
    offset_x = np.min(las.x)
    offset_y = np.min(las.y)
    offset_z = np.min(las.z)

    coords = np.vstack((
        las.x - offset_x,
        las.y - offset_y,
        las.z - offset_z
    )).T

    print(f"Applied offsets → X:{offset_x}, Y:{offset_y}, Z:{offset_z}")
    return coords


def create_point_cloud(coords, las):
    """Create Open3D point cloud with smart coloring."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # Smart coloring: RGB > Intensity
    if hasattr(las, "red"):
        print("Using RGB colors...")
        colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    elif hasattr(las, "intensity"):
        print("Using intensity mapping...")
        intensity = las.intensity / np.max(las.intensity)
        colors = np.vstack((intensity, intensity, intensity)).T
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_mesh(pcd):
    """Estimate normals and generate mesh using Poisson reconstruction."""
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )

    print("Creating mesh (Poisson reconstruction)...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    mesh.compute_vertex_normals()
    return mesh


def visualize(pcd, mesh=None):
    """Launch Open3D visualizer."""
    print("Launching visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Hackathon LiDAR Viewer",
        width=1280,
        height=720
    )

    vis.add_geometry(pcd)

    if mesh is not None:
        vis.add_geometry(mesh)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10.0, origin=[0, 0, 0]
    )
    vis.add_geometry(axes)

    vis.run()
    vis.destroy_window()


def process_lidar(file_path):
    las = load_lidar(file_path)
    coords = offset_coordinates(las)
    pcd = create_point_cloud(coords, las)
    mesh = create_mesh(pcd)
    visualize(pcd, mesh)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_lidar(sys.argv[1])
    else:
        print("Usage: python main_processor.py your_file.las")

"""

will be runned using this: 
import pdal
with open("clean_data.json") as f:
    pipeline = pdal.Pipeline(f.read())
pipeline.execute()

"""
