==========
bbox-utils
==========

Utilities to easily convert between different bounding box formats (YOLO, XYWH, XYXY, etc.).

Description
===========
You can find documentation for the project at `here <https://bbox-utils.readthedocs.io/en/latest/>`_.

**2D Bounding Box Conversions**
|Colab Image Demo|

.. |Colab Image Demo| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1NGiNhftz-hy671IHdOidkqgnaxaZ88Kn?usp=sharing

* List of points [top left, top right, bottom right, bottom left]
* XYWH: top left, width, height
* XYXY: top left, bottom right
* YOLO

**3D Bounding Box Conversions**
|Colab PCD Demo|

.. |Colab PCD Demo| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1NR9fR5hWHDtNcOyp8U0nPPjeuXR_auxd?usp=sharing

You can create a 3D bounding box with either:

* A center point, width, height, depth, and rotation
* The back-bottom-left point, width, height, depth, and rotation

You can convert between the two forms and also get a triangular polygon to use for plotting triangular meshes.

The majority of the 3D Bounding Box implementation comes from the `bbox PyPI package
<https://github.com/varunagrawal/bbox>`_.

**Visualizations**
You can use `bbox-utils` to visualize annotations within point clouds or images.

To use point clouds, you will need to install `open3d <http://www.open3d.org/docs/release/getting_started.html>`_
and `plotly <https://plotly.com/python/getting-started/>`_ with either::

    pip3 install open3d plotly==4.14.3
    pip install
    # or
    conda install -c open3d-admin open3d
    conda install -c plotly plotly=4.14.3

At the time of writing this, `open3d` requires Python < 3.9

To use images, you will need to install `OpenCV <https://opencv.org/>`_::

    pip3 install opencv-python
    # or
    conda install opencv -c conda-forge


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
