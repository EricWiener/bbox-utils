import numpy as np

from bbox_utils.utils import order_points, point_within_dimensions


class BoundingBox:
    def __init__(self, points, ordered=False):
        """Create a Bounding Box object

        Args:
            points (np.ndarray): an array of form
                                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
            ordered (bool): whether the points are already sorted
        """
        self.tl, self.tr, self.br, self.bl = (
            order_points(points) if not ordered else points
        )

    def validate_points(self, image_dimension):
        """Make sure all the bounding box points are within an image's dimensions

        Args:
            image_dimension (np.array): array with the image's dimensions
            in form (rows, cols, depth)

        Returns:
            bool: whether the points are valid
        """
        # Convert rows, cols, depth to x, y, z
        xyz_dimension = np.copy(image_dimension)
        xyz_dimension[0] = image_dimension[1]
        xyz_dimension[1] = image_dimension[0]

        tl_valid = point_within_dimensions(self.tl, xyz_dimension)
        tr_valid = point_within_dimensions(self.tr, xyz_dimension)
        br_valid = point_within_dimensions(self.br, xyz_dimension)
        bl_valid = point_within_dimensions(self.bl, xyz_dimension)

        return tl_valid and tr_valid and br_valid and bl_valid

    def to_xywh(self):
        """Returns the top-left point, width, and height

        Returns:
            tuple: tuple of form (np.array, float, float)
        """
        return self.tl.astype(int), int(self.width), int(self.height)

    @staticmethod
    def from_xywh(top_left, width, height):
        """Create a bounding box object from the top-left point, width, and height.

        Args:
            top_left (np.array): array of form [x, y]
            width (float): width of the bounding box
            height (float): height of the bounding box

        Returns:
            BoundingBox: a new bounding box instance
        """
        tl = top_left
        tr = tl + np.array([width, 0.0])
        br = tl + np.array([width, height])
        bl = tl + np.array([0.0, height])
        points = np.array([tl, tr, br, bl])
        box = BoundingBox(points, ordered=True)
        return box

    def to_xyxy(self):
        """Returns the top-left and bottom-right coordinate

        Returns:
            tuple: tuple of form (np.array, np.array)
        """
        return self.tl.astype(int), self.br.astype(int)

    @staticmethod
    def from_xyxy(top_left, bottom_right):
        """Create a bounding box object from the top-left point and bottom-right point

        Args:
            top_left (np.array): array of form [x, y]
            bottom_right ((np.array): array of form [x, y]

        Returns:
            BoundingBox: a new bounding box instance
        """
        # First make sure that the top_left and bottom_right are correctly ordered
        # We don't need to check top_left[1] < bottom_right[1]
        # because if the x-coordinate is bad, but the y-coordinate is good,
        # it will throw an assertion later
        if top_left[0] > bottom_right[0]:
            # top is below bottom, swap the points
            temp = bottom_right
            bottom_right = top_left
            top_left = temp

        # Make sure that the top_left is to left of the right point
        assert (
            top_left[1] < bottom_right[1] and top_left[0] < bottom_right[0]
        ), "Invalid xyxy points: {}, {}".format(top_left, bottom_right)

        # Make sure all points are positive
        assert (
            top_left[0] >= 0 and top_left[1] >= 0
        ), "top_left point has negative value {}".format(top_left)
        assert (
            bottom_right[0] >= 0 and bottom_right[1] >= 0
        ), "bottom_right point has negative value {}".format(bottom_right)

        t_x, t_y = top_left
        b_x, b_y = bottom_right
        height = b_y - t_y
        width = b_x - t_x
        tr = top_left + np.array([width, 0.0])
        bl = top_left + np.array([0.0, height])
        points = np.array([top_left, tr, bottom_right, bl])
        box = BoundingBox(points, ordered=True)
        return box

    def to_yolo(self, image_dimension):
        """Generates a YOLO formatted np.array with center_x, center_y, width, height

        Args:
            image_dimension (np.array): array with image dimensions of form
            (rows, cols) or (rows, cols, depth)

        Returns:
            np.array: array with YOLO formatted bounding box
        """

        # Make sure all points are within dimensions
        assert self.validate_points(
            image_dimension
        ), "Some points of the bounding box lie outside of image dimensions"

        cx, cy = self.center

        # Get normalized dimensions
        # (make sure to use rows for height and cols for width)
        img_h, img_w = image_dimension[:2]

        cx = float(cx) / img_w
        cy = float(cy) / img_h
        w = float(self.width) / img_w
        h = float(self.height) / img_h

        assert (
            0.0 <= cx <= 1.0
            and 0.0 <= cy <= 1.0
            and 0.0 <= w <= 1.0
            and 0.0 <= h <= 1.0
        ), "All YOLO values should be normalize between [0, 1]."

        return np.array([cx, cy, w, h])

    @property
    def center(self):
        """Returns the center point of the bounding box as (x, y)

        Returns:
            np.array: center of bounding box in (x, y) form. NOT (row, column) form.
        """
        x = self.tl[0] + self.width // 2
        y = self.tl[1] + self.height // 2
        return np.array([x, y]).astype(int)

    @property
    def width(self):
        return np.linalg.norm(self.tl - self.tr)

    @property
    def height(self):
        return np.linalg.norm(self.tl - self.bl)

    @property
    def points(self):
        return np.array([self.tl, self.tr, self.br, self.bl])
