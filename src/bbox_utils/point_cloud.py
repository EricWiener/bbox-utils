import numpy as np
import open3d as o3d

from bbox_utils.utils import in_google_colab


class PointCloud:
    def __init__(self, point_cloud, *args, **kwargs):
        """Create a point cloud.

        Args:
            point_cloud (obj): a valid Open3D point cloud
        """
        self.in_colab = in_google_colab()

        # Used when running tests to disable GUI
        self.display_gui = True

        if PointCloud.validate_point_cloud(point_cloud):
            self.point_cloud = point_cloud
        else:
            raise TypeError("PointCloud received invalid point cloud")

    @classmethod
    def validate_point_cloud(cls, point_cloud):
        """Validates the point cloud

        Args:
            image (obj): image to validate

        Returns:
            bool: whether the image is valid.
        """
        return isinstance(point_cloud, o3d.cpu.pybind.geometry.PointCloud)

    @classmethod
    def load_from_file(cls, file_path, *args, **kwargs):
        """Loads a point cloud from a file

        Args:
            file_path (str): the path to the file
        """
        pcd = o3d.io.read_point_cloud(file_path)
        return PointCloud(pcd)

    def display_bboxes(self, bboxes, colors="#ff0000", size=2, *args, **kwargs):
        """Display a list of bounding boxes

        Args:
            bboxes (list(BoundingBox)): a list of bounding boxes
            color (str or list(str)): a valid Plotly color:
                The 'color' property is a color and may be specified as:
                  - A hex string (e.g. '#ff0000')
                  - An rgb/rgba string (e.g. 'rgb(255,0,0)')
                  - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
                  - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
                  - A named CSS color:
                      aliceblue, antiquewhite, aqua, aquamarine, azure,
                      beige, bisque, black, blanchedalmond, blue,
                      blueviolet, brown, burlywood, cadetblue,
                      chartreuse, chocolate, coral, cornflowerblue,
                      cornsilk, crimson, cyan, darkblue, darkcyan,
                      darkgoldenrod, darkgray, darkgrey, darkgreen,
                      darkkhaki, darkmagenta, darkolivegreen, darkorange,
                      darkorchid, darkred, darksalmon, darkseagreen,
                      darkslateblue, darkslategray, darkslategrey,
                      darkturquoise, darkviolet, deeppink, deepskyblue,
                      dimgray, dimgrey, dodgerblue, firebrick,
                      floralwhite, forestgreen, fuchsia, gainsboro,
                      ghostwhite, gold, goldenrod, gray, grey, green,
                      greenyellow, honeydew, hotpink, indianred, indigo,
                      ivory, khaki, lavender, lavenderblush, lawngreen,
                      lemonchiffon, lightblue, lightcoral, lightcyan,
                      lightgoldenrodyellow, lightgray, lightgrey,
                      lightgreen, lightpink, lightsalmon, lightseagreen,
                      lightskyblue, lightslategray, lightslategrey,
                      lightsteelblue, lightyellow, lime, limegreen,
                      linen, magenta, maroon, mediumaquamarine,
                      mediumblue, mediumorchid, mediumpurple,
                      mediumseagreen, mediumslateblue, mediumspringgreen,
                      mediumturquoise, mediumvioletred, midnightblue,
                      mintcream, mistyrose, moccasin, navajowhite, navy,
                      oldlace, olive, olivedrab, orange, orangered,
                      orchid, palegoldenrod, palegreen, paleturquoise,
                      palevioletred, papayawhip, peachpuff, peru, pink,
                      plum, powderblue, purple, red, rosybrown,
                      royalblue, rebeccapurple, saddlebrown, salmon,
                      sandybrown, seagreen, seashell, sienna, silver,
                      skyblue, slateblue, slategray, slategrey, snow,
                      springgreen, steelblue, tan, teal, thistle, tomato,
                      turquoise, violet, wheat, white, whitesmoke,
                      yellow, yellowgreen
                  - A number that will be interpreted as a color
                  according to mesh3d.colorscale
        """
        # Lazy import to allow someone to use PointCloud without installing Plotly
        import plotly.express as px
        import plotly.graph_objects as go

        points = self.points

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Specify the size of the points (this needs to be a 1D array the same
        # length as x, y, and z)
        point_size = np.full(x.shape, size)

        # Plot it
        fig = px.scatter_3d(x=x, y=y, z=z, size=point_size, opacity=1)

        # The PCD scatter
        data = [go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=size))]

        # Create a list of colors if colors is just a single string
        if isinstance(colors, str):
            colors = [colors for i in range(0, len(bboxes))]

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

        if self.display_gui:  # pragma: no cover
            fig.show()

        return fig

    def display_bbox(self, bbox, color="#ff0000", size=2, *args, **kwargs):
        """Display a single bounding box

        Args:
            bbox (BoundingBox): a single bounding box
            color (string, optional): a valid Plotly color. Defaults to '#ff0000'

        Returns:
            Figure: a Plotly figure
        """
        return self.display_bboxes([bbox], [color], size)

    def display(self, size=2):
        return self.display_bboxes([], colors=[], size=2)

    @property
    def points(self):
        """Get a np.ndarray representation of the point cloud.

        Returns:
            np.ndarray: the point cloud's points
        """
        return np.asarray(self.point_cloud.points)

    @property
    def number_of_points(self):
        """The number of points within a point cloud.

        Returns:
            int: the number of points within a point cloud.
        """
        return len(self.points)

    def crop(self, bbox):
        """Extract a point cloud from a 3D bounding box.

        Source: https://stackoverflow.com/a/65350251/6942666

        Args:
            bbox (BoundingBox3D): a 3D bounding box

        Returns:
            PointCloud: a new point cloud with just the points within
                the bounding box.
        """
        # Convert the corners array to have type float64
        bounding_polygon = bbox.p.astype("float64")

        # Create a SelectionPolygonVolume
        vol = o3d.visualization.SelectionPolygonVolume()

        # You need to specify what axis to orient the polygon to.
        # I choose the "Y" axis. I made the max value the maximum Y of
        # the polygon vertices and the min value the minimum Y of the
        # polygon vertices.
        vol.orthogonal_axis = "Y"
        vol.axis_max = np.max(bounding_polygon[:, 1])
        vol.axis_min = np.min(bounding_polygon[:, 1])

        # Set all the Y values to 0 (they aren't needed since we specified what they
        # should be using just vol.axis_max and vol.axis_min).
        bounding_polygon[:, 1] = 0

        # Convert the np.array to a Vector3dVector
        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

        # Crop the point cloud using the Vector3dVector
        cropped_pcd = vol.crop_point_cloud(self.point_cloud)

        return PointCloud(cropped_pcd)
