import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .visualizer_3d import Visualizer3D


class Visualizer3DPlotly(Visualizer3D):
    def display_bboxes(self, bboxes, colors, size=2, *args, **kwargs):
        """Display a list of bounding boxes

        Args:
            bboxes (list(BoundingBox)): a list of bounding boxes
            color (str or list(str)): a list of colors for each bounding box.
                Color should be specified in BGR.
        """
        points = self.point_cloud.points

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Specify the size of the points (this needs to be a 1D array the same
        # length as x, y, and z)
        point_size = np.full(x.shape, size)

        # Plot it
        fig = px.scatter_3d(x=x, y=y, z=z, size=point_size, opacity=1)

        # The PCD scatter
        data = [go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=size))]

        # Add any annotations
        for idx, bbox in enumerate(bboxes):
            # Get corners and triangle vertices
            corners, triangle_vertices = bbox.p, bbox.triangle_vertices

            data.append(
                go.Mesh3d(
                    x=corners[:, 0],
                    y=corners[:, 1],
                    z=corners[:, 2],
                    i=triangle_vertices[0],
                    j=triangle_vertices[1],
                    k=triangle_vertices[2],
                    opacity=0.6,
                    color=colors[idx],
                    flatshading=True,
                )
            )

        fig = go.Figure(data=data)
        fig.show()

    def display_bbox(self, bbox, color=(0, 0, 255), size=2, *args, **kwargs):
        """Display a single bounding box

        Args:
            bbox (BoundingBox): a single bounding box
            color (tuple, optional): color of the bounding box in BGR.
                Defaults to (0, 0, 255).
        """
        self.display_bboxes([bbox], [color], size)
