import cv2
import numpy as np

from .visualizer_2d import Visualizer2D


class VisualizerCV2(Visualizer2D):
    def __init__(self, image):
        super(VisualizerCV2, self).__init__(image)

    @classmethod
    def validate_image(cls, image):
        # @TODO: I'm not sure this is going to be called
        if isinstance(image, np.ndarray):
            return True
        else:
            raise TypeError("Visualizer_CV2 must be initialized with a np.ndarry")

    @classmethod
    def load_from_file(cls, file_path):
        """Loads an image from a file

        Args:
            file_path (str): the path to the file

        Returns:
            VisualizerCV2: an instance of VisualizerCV2
        """
        image = cv2.imread(file_path)
        return VisualizerCV2(image)

    def display_bbox(self, bbox, color=(0, 0, 255)):
        self.display_bboxes([bbox], [color])

    def display_bboxes(self, bboxes, colors):
        for idx, bbox in enumerate(bboxes):
            xy1, xy2 = bbox.to_xyxy()
            cv2.rectangle(self.image, xy1, xy2, colors[idx], 2)

        # Display image correctly in Google Colab
        if not self.in_colab:
            cv2.imshow(self.image, "Bounding Boxes")
        else:
            from google.colab.patches import cv2_imshow

            cv2_imshow(self.image)
