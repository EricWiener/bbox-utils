import numpy as np
import pytest

from bbox_utils import BoundingBox3D
from bbox_utils.point_cloud import PointCloud


def test_point_cloud_load_from_file():
    # Load a PCD and make sure we can count the number of points
    pcd = PointCloud.load_from_file("./tests/assets/demo_pointcloud.pcd")
    assert pcd.number_of_points == 2442


def create_bounding_box():
    values = {
        "position": {
            "x": 6.536307742091793,
            "y": -2.1117338632172125,
            "z": 1.1549238563313258,
        },
        "rotation": {
            "x": -0.08282295820270243,
            "y": -0.10018696688968483,
            "z": -0.6927483959297558,
        },
        "dimensions": {
            "x": 2.3551822605229615,
            "y": 2.075954356460532,
            "z": 0.7333401523302493,
        },
    }

    center = np.array(
        [values["position"]["x"], values["position"]["y"], values["position"]["z"]]
    )

    dimensions = np.array(
        [
            values["dimensions"]["x"],
            values["dimensions"]["y"],
            values["dimensions"]["z"],
        ]
    )

    rotation = np.array(
        [values["rotation"]["x"], values["rotation"]["y"], values["rotation"]["z"]]
    )

    bbox = BoundingBox3D.from_center_dimension_euler(
        center=center, dimension=dimensions, euler_angles=rotation
    )
    return bbox


def test_point_cloud_crop():
    pcd = PointCloud.load_from_file("./tests/assets/demo_pointcloud.pcd")
    bbox = create_bounding_box()
    annotation_pcd = pcd.crop(bbox)
    assert annotation_pcd.number_of_points == 124


def test_point_cloud_constructor_invalid():
    with pytest.raises(TypeError):
        _ = PointCloud("no bueno")

    with pytest.raises(TypeError):
        _ = PointCloud(5)

    with pytest.raises(TypeError):
        _ = PointCloud(np.array([1, 2, 3]))


def test_point_cloud_constructor_valid():
    pcd = PointCloud.load_from_file("./tests/assets/demo_pointcloud.pcd")
    pcd_two = PointCloud(pcd.point_cloud)
    assert pcd_two.number_of_points == 2442


def test_point_cloud_display():
    pcd = PointCloud.load_from_file("./tests/assets/demo_pointcloud.pcd")
    pcd.display_gui = False
    fig = pcd.display()

    # @TODO: perform a better test for the figure
    assert fig is not None


def test_point_cloud_display_bbox():
    pcd = PointCloud.load_from_file("./tests/assets/demo_pointcloud.pcd")
    pcd.display_gui = False
    bbox = create_bounding_box()
    fig = pcd.display_bbox(bbox)

    # @TODO: perform a better test for the figure
    assert fig is not None


def test_point_cloud_display_bboxes_single_box():
    pcd = PointCloud.load_from_file("./tests/assets/demo_pointcloud.pcd")
    pcd.display_gui = False
    bbox = create_bounding_box()
    fig = pcd.display_bboxes([bbox])

    # @TODO: perform a better test for the figure
    assert fig is not None
