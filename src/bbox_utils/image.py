import cv2
import numpy as np

from bbox_utils.utils import in_google_colab


class Image:
    def __init__(self, image, *args, **kwargs):
        """Create a Visualizer.

        Args:
            image (obj): a valid image object
        """
        self.in_colab = in_google_colab()

        if Image.validate_image(image):
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
        if isinstance(image, np.ndarray):
            return True
        else:
            raise TypeError("Visualizer_CV2 must be initialized with a np.ndarry")

    @classmethod
    def load_from_file(cls, file_path, *args, **kwargs):
        """Loads an image from a file

        Args:
            file_path (str): the path to the file
        """
        image = cv2.imread(file_path)
        return Image(image)

    def display_bboxes(self, bboxes, colors, *args, **kwargs):
        """Display a list of bounding boxes

        Args:
            bboxes (list(BoundingBox)): a list of bounding boxes
            color (str or list(str)): a list of colors for each bounding box.
                Color should be specified in BGR.
        """
        for idx, bbox in enumerate(bboxes):
            xy1, xy2 = bbox.to_xyxy()
            cv2.rectangle(self.image, xy1, xy2, colors[idx], 2)

        # Display image correctly in Google Colab
        if not self.in_colab:
            cv2.imshow(self.image, "Bounding Boxes")
        else:
            from google.colab.patches import cv2_imshow

            cv2_imshow(self.image)

    def display_bbox(self, bbox, color=(0, 0, 255), *args, **kwargs):
        """Display a single bounding box

        Args:
            bbox (BoundingBox): a single bounding box
            color (tuple, optional): color of the bounding box in BGR.
                Defaults to (0, 0, 255).
        """
        self.display_bboxes([bbox], [color])
