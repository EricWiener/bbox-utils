import numpy as np
import pytest
from PIL import Image, ImageDraw

from bbox_utils import BoundingBox3D


class TestBoundingBox3D:
    @classmethod
    def setup_class(cls):
        # sample cuboid
        cuboid = {
            "center": {
                "x": -49.19743041908411,
                "y": 12.38666074615689,
                "z": 0.782056864653507,
            },
            "dimensions": {
                "length": 5.340892485711914,
                "width": 2.457703972075464,
                "height": 1.9422248281533563,
            },
            "rotation": {
                "w": 0.9997472337219893,
                "x": 0.0,
                "y": 0.0,
                "z": 0.022482630300529462,
            },
        }
        center = cuboid["center"]
        dim = cuboid["dimensions"]
        rotation = cuboid["rotation"]

        cls.box = BoundingBox3D(
            center["x"],
            center["y"],
            center["z"],
            length=dim["length"],
            width=dim["width"],
            height=dim["height"],
            rw=rotation["w"],
            rx=rotation["x"],
            ry=rotation["y"],
            rz=rotation["z"],
        )
        cls.cuboid = cuboid

    def test_points(self):
        points = np.array(
            [
                [-51.80993533, 11.03900409, -0.18905555],
                [-51.92041869, 13.49422348, -0.18905555],
                [-46.58492551, 13.7343174, -0.18905555],
                [-46.47444215, 11.27909801, -0.18905555],
                [-51.80993533, 11.03900409, 1.75316928],
                [-51.92041869, 13.49422348, 1.75316928],
                [-46.58492551, 13.7343174, 1.75316928],
                [-46.47444215, 11.27909801, 1.75316928],
            ]
        )

        assert np.allclose(self.box.p1, points[0, :])
        assert np.allclose(self.box.p2, points[1, :])
        assert np.allclose(self.box.p3, points[2, :])
        assert np.allclose(self.box.p4, points[3, :])
        assert np.allclose(self.box.p5, points[4, :])
        assert np.allclose(self.box.p6, points[5, :])
        assert np.allclose(self.box.p7, points[6, :])
        assert np.allclose(self.box.p8, points[7, :])

        assert np.allclose(self.box.p, points)

    def test_center(self):
        center = np.array([-49.19743041908411, 12.38666074615689, 0.782056864653507])
        assert np.array_equal(self.box.center, center)

    def test_center_points(self):
        assert self.box.cx == self.cuboid["center"]["x"]
        assert self.box.cy == self.cuboid["center"]["y"]
        assert self.box.cz == self.cuboid["center"]["z"]

    def test_init_center(self):
        box = BoundingBox3D(*[self.box.cx, self.box.cy, self.box.cz], is_center=True)
        assert np.array_equal(box.center, self.box.center)

    def test_init_non_center(self):
        cx, cy, cz = self.box.cx, self.box.cy, self.box.cz
        box = BoundingBox3D(
            cx - self.box.length / 2,
            cy - self.box.width / 2,
            cz - self.box.height / 2,
            self.box.length,
            self.box.width,
            self.box.height,
            is_center=False,
        )
        assert np.array_equal(box.center, self.box.center)

    def test_center_setter(self):
        box = self.box.copy()
        box.center = np.asarray([0, 1, 2])
        assert np.array_equal(box.center, np.asarray([0, 1, 2]))

        with pytest.raises(ValueError):
            box.center = np.asarray([0, 1])
            box.center = np.asarray([0, 1, 2, 3])

    def test_dimensions(self):
        assert self.box.length == self.cuboid["dimensions"]["length"]
        assert self.box.width == self.cuboid["dimensions"]["width"]
        assert self.box.height == self.cuboid["dimensions"]["height"]

    def test_quaternion(self):
        q = np.array([0.9997472337219893, 0.0, 0.0, 0.022482630300529462])
        assert np.array_equal(self.box.q, q)
        # alternative attribute for the quaternion
        assert np.array_equal(self.box.quaternion, q)

        box = BoundingBox3D(self.box.cx, self.box.cy, self.box.cz, q=self.box.q)
        assert np.array_equal(self.box.q, box.q)

    def test_euler_angles(self):
        box = BoundingBox3D(
            3.163,
            z=2.468,
            y=34.677,
            height=1.529,
            width=1.587,
            length=3.948,
            euler_angles=[0, 0, -1.59],
        )
        q = np.array([0.7002847660410397, 0.0, 0.0, -0.713863604935036])
        assert np.allclose(box.q, q)

    def test_projection(self):
        K = np.array(
            [
                [1406.3359, 0.0, 966.366034, 0.0],
                [0.0, 1408.94297, 607.479746, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        R = np.array(
            [
                [0.50478576, 0.86323317, -0.00445338],
                [-0.00422247, -0.00268975, -0.99998747],
                [-0.86323433, 0.50479824, 0.00228723],
            ]
        )

        t = np.array([[-0.75116634], [1.35776453], [0.87137971]])

        u = np.empty((8, 2))
        u[0] = self.project(self.box.p1, K, R, t)
        u[1] = self.project(self.box.p2, K, R, t)
        u[2] = self.project(self.box.p3, K, R, t)
        u[3] = self.project(self.box.p4, K, R, t)
        u[4] = self.project(self.box.p5, K, R, t)
        u[5] = self.project(self.box.p6, K, R, t)
        u[6] = self.project(self.box.p7, K, R, t)
        u[7] = self.project(self.box.p8, K, R, t)

        image_points = np.array(
            [
                [488.84269983, 655.2790429],
                [556.26022377, 653.89914093],
                [602.90920984, 657.55445187],
                [530.3490365, 659.17142666],
                [488.64644505, 601.79933826],
                [556.06325411, 601.7790501],
                [602.68953074, 600.56675254],
                [530.12998103, 600.55433066],
            ]
        )

        for i in range(8):
            assert np.allclose(u[i], image_points[i])

    def test_setters(self):
        x, y, z, height, width, length = np.random.rand(6)
        box = self.box.copy()
        box.cx = x
        box.cy = y
        box.cz = z
        box.height = height
        box.width = width
        box.length = length
        assert box.cx == x
        assert box.cy == y
        assert box.cz == z
        assert box.height == height
        assert box.width == width
        assert box.length == length

        q = np.random.rand(4)
        box.q = q
        assert np.array_equal(box.q, q)
        q = np.random.rand(4)
        box.quaternion = q
        assert np.array_equal(box.quaternion, q)

        center = np.random.rand(3)
        box.center = center
        assert np.array_equal(box.center, center)

    def test_bad_setters(self):
        inputs = [[1, 2], np.zeros((3)), np.zeros((3, 1)), np.zeros((1, 3)), "center"]
        for x in inputs:
            with pytest.raises((ValueError, TypeError)):
                self.box.cx = x
            with pytest.raises((ValueError, TypeError)):
                self.box.cy = x
            with pytest.raises((ValueError, TypeError)):
                self.box.cz = x
            with pytest.raises((ValueError, TypeError)):
                self.box.height = x
            with pytest.raises((ValueError, TypeError)):
                self.box.width = x
            with pytest.raises((ValueError, TypeError)):
                self.box.length = x

        quaternion_inputs = [1, [1], [1, 2, 3], [1, 2, 3, 4, 5], "quaternion"]
        for q in quaternion_inputs:
            with pytest.raises((ValueError, TypeError)):
                self.box.q = q

    def test_render(self):
        K = np.array(
            [
                [1406.3359, 0.0, 966.366034, 0.0],
                [0.0, 1408.94297, 607.479746, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        R = np.array(
            [
                [0.50478576, 0.86323317, -0.00445338],
                [-0.00422247, -0.00268975, -0.99998747],
                [-0.86323433, 0.50479824, 0.00228723],
            ]
        )

        t = np.array([[-0.75116634], [1.35776453], [0.87137971]])

        u = np.empty((8, 2))
        u[0] = self.project(self.box.p1, K, R, t)
        u[1] = self.project(self.box.p2, K, R, t)
        u[2] = self.project(self.box.p3, K, R, t)
        u[3] = self.project(self.box.p4, K, R, t)
        u[4] = self.project(self.box.p5, K, R, t)
        u[5] = self.project(self.box.p6, K, R, t)
        u[6] = self.project(self.box.p7, K, R, t)
        u[7] = self.project(self.box.p8, K, R, t)

        img = Image.new(mode="RGB", size=(512, 512))

        dist_coeff = [-0.17120984449230167, 0.1256910189977147, -0.029726711792577232]

        for i in range(u.shape[0]):
            u[i] = self.distortion_correction(u[i], dist_coeff, img.size)

        img = self.draw_cuboid(img, u)
        # img.show()

    @staticmethod
    def project(p, K, R, t):
        p = np.hstack((p, 1))
        E = np.eye(4)
        E[0:3, 0:3] = R
        E[0:3, 3:4] = t

        u = K @ E @ p
        # print("projection", u)
        return u[0:2] / u[2]

    @staticmethod
    def distortion_correction(u, dist_coeff, img_size):
        w, h = img_size
        # normalize the image coords
        x = 2 * u[0] / w - 1
        y = 2 * u[1] / h - 1
        r = np.linalg.norm(np.array(x, y))

        r2 = r ** 2

        distortion = 0
        for i in range(len(dist_coeff)):
            distortion += (r2 ** (i + 1)) * dist_coeff[i]

        # correct for distortion
        v = u + (u - np.array([w, h]) / 2) * distortion
        return v

    @staticmethod
    def draw_cuboid(img, p, color=None):
        draw = ImageDraw.Draw(img)
        color = color or tuple(np.random.choice(range(256), size=3))

        draw.line([p[0][0], p[0][1], p[1][0], p[1][1]], fill=color, width=2)
        draw.line([p[1][0], p[1][1], p[5][0], p[5][1]], fill=color, width=2)
        draw.line([p[5][0], p[5][1], p[4][0], p[4][1]], fill=color, width=2)
        draw.line([p[4][0], p[4][1], p[0][0], p[0][1]], fill=color, width=2)

        draw.line([p[3][0], p[3][1], p[2][0], p[2][1]], fill=color, width=2)
        draw.line([p[2][0], p[2][1], p[6][0], p[6][1]], fill=color, width=2)
        draw.line([p[6][0], p[6][1], p[7][0], p[7][1]], fill=color, width=2)
        draw.line([p[7][0], p[7][1], p[3][0], p[3][1]], fill=color, width=2)

        draw.line([p[0][0], p[0][1], p[3][0], p[3][1]], fill=color, width=2)
        draw.line([p[1][0], p[1][1], p[2][0], p[2][1]], fill=color, width=2)
        draw.line([p[5][0], p[5][1], p[6][0], p[6][1]], fill=color, width=2)
        draw.line([p[4][0], p[4][1], p[7][0], p[7][1]], fill=color, width=2)
        return img

    def test_repr(self):
        representation = (
            "BoundingBox3D(x=-49.19743041908411, y=12.38666074615689, "
            "z=0.782056864653507), "
            "length=5.340892485711914, width=2.457703972075464, "
            "height=1.9422248281533563, "
            "q=(0.9997472337219893, 0.0, 0.0, 0.022482630300529462))"
        )
        assert repr(self.box) == representation

    def test_triangle_vertices(self):
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        triangle_vertices = np.vstack([i, j, k])
        assert np.array_equal(self.box.triangle_vertices, triangle_vertices)

    def test_edges(self):
        edges = np.array(
            [
                [0, 1],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 5],
                [2, 3],
                [2, 6],
                [3, 7],
                [4, 5],
                [4, 7],
                [5, 6],
                [6, 7],
            ]
        )
        assert np.array_equal(self.box.edges, edges)


def test_from_xyz_xyz():
    xyz1 = [10, 0, 0]
    xyz2 = [50, 60, 70]

    # Test with smallest, largest
    bbox = BoundingBox3D.from_xyzxyz(xyz1, xyz2)
    assert np.allclose(bbox.center, np.array([30, 30, 35]))

    # Test with largest, smallest point
    # The order of the points should not matter
    bbox = BoundingBox3D.from_xyzxyz(xyz2, xyz1)
    assert np.allclose(bbox.center, np.array([30, 30, 35]))

    # Make sure the dimensions are correct
    assert bbox.length == 40
    assert bbox.width == 60
    assert bbox.height == 70

    # Test with a different set of opposite points
    # No longer absolute min and absolute max
    xyz1 = [10, 0, 70]
    xyz2 = [50, 60, 0]
    bbox = BoundingBox3D.from_xyzxyz(xyz2, xyz1)
    assert np.allclose(bbox.center, np.array([30, 30, 35]))
