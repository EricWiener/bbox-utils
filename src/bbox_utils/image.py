import cv2
import numpy as np


class Image:
    def __init__(self, image, *args, **kwargs):
        """Create a Visualizer.

        Args:
            image (obj): a valid image object
        """
        if Image.validate_image(image):
            self.image = image

    @classmethod
    def validate_image(cls, image):
        """Validate an image object

        Args:
            image (obj): image to validate

        Returns:
            bool: whether the image is valid.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be initialized with a np.ndarry")

        if image.ndim < 2:
            raise ValueError(
                "Image must be initialized with a np.ndarray with >= 2 dimensions"
            )

        return True

    @classmethod
    def load_from_file(cls, file_path, *args, **kwargs):
        """Loads an image from a file

        Args:
            file_path (str): the path to the file
        """
        image = cv2.imread(file_path)
        return Image(image)

    def display_bboxes(self, bboxes, colors, *args, **kwargs):  # pragma: no cover
        """Display a list of bounding boxes

        Args:
            bboxes (list(BoundingBox)): a list of bounding boxes
            color (str or list(str)): a list of colors for each bounding box.
                Color should be specified in BGR.
        """
        image = self.image.copy()

        for idx, bbox in enumerate(bboxes):
            xy1, xy2 = bbox.to_xyxy()
            cv2.rectangle(image, tuple(xy1), tuple(xy2), colors[idx], 2)

        self.display(image)

    def display_bbox(
        self, bbox, color=(0, 0, 255), *args, **kwargs
    ):  # pragma: no cover
        """Display a single bounding box

        Args:
            bbox (BoundingBox): a single bounding box
            color (tuple, optional): color of the bounding box in BGR.
                Defaults to (0, 0, 255).
        """
        self.display_bboxes([bbox], [color])

    def display(self, image=None, title=None, library="matplotlib"):  # pragma: no cover
        """Display an image using a library of your choice.

        Args:
            image (np.ndarray, optional): the image to display.
                If none specified, uses self.image.
            title (str, optional): the title to use. Defaults to None.
            library (str, optional): the library to use to display the image.
                Defaults to "matplotlib". Can also choose "opencv".

        Raises:
            ValueError: an error if an invalid library argument is passed.
        """

        # This function can be called from display_bbox.
        # We don't directly modify self.image when drawing bounding boxes,
        # so we allow an image to be passed (that isn't self.image)
        if image is None:
            image = self.image

        if library == "matplotlib":
            Image.display_matplotlib(image, title)
        elif library == "opencv":
            Image.display_opencv(image, title)
        else:
            raise ValueError(
                "Library argument to Image.display() not recognized: {}".format(library)
            )

    @classmethod
    def display_matplotlib(cls, image, title=None):  # pragma: no cover
        """Display an image using matplotlib.

        Args:
            image (np.ndarray): a numpy ndarray in BGR format.
            title (str, optional): the title to give the plot. Defaults to None.
        """
        import matplotlib.pyplot as plt

        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if title:
            plt.title(title)
        plt.show()

    @classmethod
    def display_cv2(cls, image, title=None):  # pragma: no cover
        """Display an image using OpenCV.

        Args:
            image (np.ndarray): a numpy ndarray in BGR format.
            title (str, optional): the title to give the plot. Defaults to None.
        """
        from bbox_utils.utils import in_google_colab

        in_colab = in_google_colab()

        # Display image correctly in Google Colab
        if not in_colab:
            cv2.imshow(title, image)
        else:
            from google.colab.patches import cv2_imshow

            cv2_imshow(image)
