import numpy as np

from bbox_utils.utils import order_points


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
        tr = tl + np.array([0.0, width])
        br = tl + np.array([height, width])
        bl = tl + np.array([height, 0.0])
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

        tl = top_left
        br = bottom_right
        height = br[0] - tl[0]
        width = br[1] - tl[1]
        tr = top_left + np.array([0.0, width])
        bl = top_left + np.array([height, 0.0])
        points = np.array([tl, tr, br, bl])
        box = BoundingBox(points, ordered=True)
        return box

    def to_yolo(self, image_dimension):
        cx, cy = self.center

        # Get normalized dimensions
        img_h, img_w = image_dimension[:2]

        cx = float(cx) / img_w
        cy = float(cy) / img_h
        w = float(self.width) / img_w
        h = float(self.height) / img_h
        return np.array([cx, cy, w, h])

    @property
    def center(self):
        cx = self.tl[0] + self.width // 2
        cy = self.tl[1] + self.height // 2
        return np.array([cx, cy]).astype(int)

    @property
    def width(self):
        return np.linalg.norm(self.tl - self.tr)

    @property
    def height(self):
        return np.linalg.norm(self.tl - self.bl)

    @property
    def points(self):
        return np.array([self.tl, self.tr, self.br, self.bl])
