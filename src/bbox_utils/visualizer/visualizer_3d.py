from abc import ABC

import open3d as o3d

from bbox_utils.utils import in_google_colab


class Visualizer3D(ABC):
    def __init__(self, point_cloud, *args, **kwargs):
        """Create a 3D Visualizer.

        Args:
            image (obj): a valid image object
        """
        self.in_colab = in_google_colab()

        if Visualizer3D.validate_point_cloud(point_cloud):
            self.point_cloud = point_cloud
        else:
            raise TypeError("Visualizer3D received invalid point cloud")

    @classmethod
    def validate_point_cloud(cls, point_cloud):
        """Validates the point cloud

        Args:
            image (obj): image to validate

        Returns:
            bool: whether the image is valid.
        """
        return True

    @classmethod
    def load_from_file(cls, file_path, *args, **kwargs):
        """Loads a point cloud from a file

        Args:
            file_path (str): the path to the file
        """
        pcd = o3d.io.read_point_cloud(file_path)
        return Visualizer3D(pcd)

    def display_bboxes(self, bboxes, colors, *args, **kwargs):
        """Display a list of bounding boxes

        Args:
            bboxes (list(BoundingBox)): a list of bounding boxes
            color (str or list(str)): a list of colors for each bounding box.
                Color should be specified in BGR.
        """
        pass

    def display_bbox(self, bbox, color=(0, 0, 255), *args, **kwargs):
        """Display a single bounding box

        Args:
            bbox (BoundingBox): a single bounding box
            color (tuple, optional): color of the bounding box in BGR.
                Defaults to (0, 0, 255).
        """
        pass
