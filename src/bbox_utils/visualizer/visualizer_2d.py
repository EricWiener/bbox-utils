from abc import ABC

from bbox_utils.utils import in_google_colab


class Visualizer2D(ABC):
    def __init__(self, image, *args, **kwargs):
        """Create a Visualizer.

        Args:
            image (obj): a valid image object
        """
        self.in_colab = in_google_colab()

        if Visualizer2D.validate_image(image):
            self.image = image
        else:
            raise TypeError("Visualizer2D received invalid image")

    @classmethod
    def validate_image(cls, image):
        """Validate an image object

        Args:
            image (obj): image to validate

        Returns:
            bool: whether the image is valid.
        """
        return True

    @classmethod
    def load_from_file(cls, file_path, *args, **kwargs):
        """Loads an image from a file

        Args:
            file_path (str): the path to the file
        """
        pass

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
