==========
bbox-utils
==========

Utilities to easily convert between different bounding box formats (YOLO, XYWH, XYXY, etc.).

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

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd bbox-utils
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate


.. _pre-commit: http://pre-commit.com/

Note
====

This project has been set up using PyScaffold 4.0rc1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
