from bbox_utils.point_cloud import PointCloud


def test_point_cloud_load_from_file():
    # Load a PCD and make sure we can count the number of points
    pcd = PointCloud.load_from_file("./tests/assets/demo_pointcloud.pcd")
    assert pcd.number_of_points == 2442
