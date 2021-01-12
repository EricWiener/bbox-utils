import numpy as np


def pointwise_distance(pts1, pts2):
    """Calculates the distance between pairs of points

    Args:
        pts1 (np.ndarray): array of form [[x1, y1], [x2, y2], ...]
        pts2 (np.ndarray): array of form [[x1, y1], [x2, y2], ...]

    Returns:
        np.array: distances between corresponding points
    """
    dist = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))
    return dist


def order_points(pts):
    """Orders points in form [top left, top right, bottom right, bottom left].
    Source:
    https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

    Args:
        pts (np.ndarray): list of points of form
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        [type]: [description]
    """
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point. Note: this is a valid assumption because
    # we are dealing with rectangles only.
    # We need to use this instead of just using min/max to handle the case where
    # there are points that have the same x or y value.
    D = pointwise_distance(np.vstack([tl, tl]), right_most)

    br, tr = right_most[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def point_within_dimensions(point, image_dimensions):
    """Checks to see if a point falls inside an image's dimension.
    Works for any number of dimensions. Acceptable range is [0, dim)

    Args:
        point (np.array): array with the point's coordinates
        image_dimensions (np.array): array with the image dimensions

    Returns:
        bool: whether the point lies within the dimensions
    """
    assert len(point) == len(
        image_dimensions
    ), "Point dimensions {} doesn't equal image dimension {}".format(
        len(point), len(image_dimensions)
    )

    within_bounds = True
    for i, val in enumerate(point):
        within_bounds = within_bounds and 0 <= val < image_dimensions[i]

    return within_bounds
