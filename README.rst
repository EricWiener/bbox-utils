.. image:: https://travis-ci.com/EricWiener/bbox-utils.svg?branch=master
    :target: https://travis-ci.com/EricWiener/bbox-utils

.. image:: https://coveralls.io/repos/github/EricWiener/bbox-utils/badge.svg?branch=master
    :target: https://coveralls.io/github/EricWiener/bbox-utils?branch=master

.. image:: https://img.shields.io/pypi/v/bbox-utils
    :alt: PyPI

==========
bbox-utils
==========


Utilities to easily convert between different bounding box formats (YOLO, XYWH, XYXY, etc.).

You can install bbox-utils with PyPI: ``$ pip install bbox-utils``.

Description
===========
You can find documentation for the project at `here <https://bbox-utils.readthedocs.io/en/latest/>`_.

**2D Bounding Box Conversions**

- List of points [top left, top right, bottom right, bottom left]
- XYWH: top left, width, height
- XYXY: top left, bottom right
- YOLO

**3D Bounding Box Conversions**
You can create a 3D bounding box with either:

- A center point, width, height, depth, and rotation
- The eight vertices

You can convert between the two forms and also get a
triangular polygon to use for plotting triangular meshes.
